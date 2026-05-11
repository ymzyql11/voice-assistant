import asyncio
import os
import sys
import queue
import math
import time
import numpy as np
import sounddevice as sd
import sherpa_onnx
from openai import AsyncOpenAI

# ================= 核心配置 =================
QWEN_API_KEY = "sk-3aa12f09ba6d4a98a89f16aa36875e9e"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ASR_TOKENS = "./model/tokens.txt"
ASR_ENCODER = "./model/encoder.int8.onnx"
ASR_DECODER = "./model/decoder.int8.onnx"

class PipecatLiteAssistant:
    def __init__(self):
        self.sample_rate = 16000
        self.client = AsyncOpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
        
        # 异步管道队列 (容量限制防溢出)
        self.audio_q = queue.Queue(maxsize=32)
        self.text_q = asyncio.Queue(maxsize=3)
        
        # 状态管理
        self.history = [{"role": "system", "content": "你是一个专业的语音助手。回答简洁直接，避免冗长铺垫。"}]
        self.rms_threshold = 0.008          # 语音能量门控
        self.silence_timeout = 2.5          # 静音强制断句(秒)
        self.last_voice_time = time.time()
        
        # 任务句柄 (用于中断)
        self.llm_task = None
        self.interrupt_event = asyncio.Event()
        
        self.init_asr()
        self.init_audio()

    def init_asr(self):
        if not all(os.path.exists(f) for f in [ASR_TOKENS, ASR_ENCODER, ASR_DECODER]):
            print("❌ 模型文件缺失，请检查路径。")
            sys.exit(1)
            
        print("⏳ 加载语音引擎 (Paraformer) ...")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
            encoder=ASR_ENCODER, decoder=ASR_DECODER, tokens=ASR_TOKENS,
            num_threads=4, enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.5, rule2_min_trailing_silence=1.2,
        )
        self.stream = self.recognizer.create_stream()

    def init_audio(self):
        devices = sd.query_devices()
        idx = sd.default.device[0]
        if idx is None:
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0 and 'Mix' not in d['name']:
                    idx = i; break
        self.device_idx = idx
        print(f"🎤 已绑定麦克风: {devices[self.device_idx]['name']}")

    def audio_callback(self, indata, frames, time_info, status):
        """声卡回调：极速入队，绝不阻塞"""
        if status: print(f"⚠️ 音频警告: {status}")
        try:
            # RMS 计算 (向量化加速)
            rms = math.sqrt(np.mean(indata**2))
            if rms > self.rms_threshold:
                self.last_voice_time = time.time()
            self.audio_q.put_nowait(indata.copy().ravel())
        except queue.Full:
            pass  # 满则丢弃，保护实时性

    async def asr_pipeline(self):
        """ASR 处理节点：连续解码 + 智能端点检测"""
        while True:
            chunk = await asyncio.to_thread(self.audio_q.get)
            self.stream.accept_waveform(self.sample_rate, chunk)
            
            # 异步执行 CPU 密集型解码
            while self.recognizer.is_ready(self.stream):
                await asyncio.to_thread(self.recognizer.decode_stream, self.stream)
                
            # 触发条件：正常端点 或 超时强制断句
            if self.recognizer.is_endpoint(self.stream) or \
               (time.time() - self.last_voice_time > self.silence_timeout and self.recognizer.is_ready(self.stream)):
                
                text = self.recognizer.get_result(self.stream).strip()
                clean = text if len(text) > 2 and not text.replace("。","").strip() in ["嗯","啊","哦"] else None
                
                if clean:
                    print(f"\n✅ 识别: {clean}")
                    await self.text_q.put(clean)
                    
                    # 触发中断信号
                    if self.llm_task and not self.llm_task.done():
                        self.llm_task.cancel()
                        await asyncio.sleep(0.05)
                        
                self.recognizer.reset(self.stream)

    async def llm_pipeline(self):
        """LLM 处理节点：流式生成 + 安全中断"""
        while True:
            user_text = await self.text_q.get()
            
            # 历史剪枝 (System + 最近3轮)
            if len(self.history) > 7:
                self.history = [self.history[0]] + self.history[-6:]
            self.history.append({"role": "user", "content": user_text})
            
            print("🌐 思考中...")
            try:
                stream = await self.client.chat.completions.create(
                    model="qwen-plus", messages=self.history, stream=True,
                    extra_body={"enable_search": True}
                )
                
                print("🤖: ", end="", flush=True)
                full_reply = ""
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        print(content, end="", flush=True)
                        full_reply += content
                print("\n")
                
                if full_reply:
                    self.history.append({"role": "assistant", "content": full_reply})
                    
            except asyncio.CancelledError:
                print("\n🛑 [已打断]")
            except Exception as e:
                print(f"\n❌ 请求失败: {e}")
            finally:
                self.llm_task = None

    async def run(self):
        print("="*42)
        print("🎙️ PipecatLite 语音助手已就绪")
        print("💡 架构: asyncio 管道 | 延迟优化 | 内存安全")
        print("="*42)
        
        # 并行启动管道节点
        asr_task = asyncio.create_task(self.asr_pipeline())
        llm_task = asyncio.create_task(self.llm_pipeline())
        
        # 保持主循环
        try:
            with sd.InputStream(device=self.device_idx, samplerate=self.sample_rate, 
                                channels=1, dtype='float32', callback=self.audio_callback):
                while True:
                    await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 安全退出。")
            asr_task.cancel(); llm_task.cancel()
            await asyncio.gather(asr_task, llm_task, return_exceptions=True)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    asyncio.run(PipecatLiteAssistant().run())
