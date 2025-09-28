# tradutor.py - Tradução de áudio em tempo real (console)

# --- 1. Importações ---
import sounddevice as sd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 'faster-whisper' é uma implementação otimizada do Whisper
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("AVISO: Instale 'faster-whisper' e 'openvino' com: pip install faster-whisper openvino")
    exit()
import queue      # Fila para comunicação entre threads
import threading  # Processamento em segundo plano
import os         # Limpeza de tela

# --- 2. Configurações ---
# Áudio
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'float32'
BLOCK_SIZE = 1024  # Tamanho do bloco para o callback de áudio

# Processamento (otimizado para baixa latência)
CHUNK_SECONDS = 3  # Tamanho do chunk de áudio para processamento (em segundos)
CHUNK_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE)
SLIDE_SECONDS = 1  # Sobreposição da janela deslizante (em segundos)
SLIDE_SAMPLES = int(SLIDE_SECONDS * SAMPLE_RATE)

# Modelos
WHISPER_MODEL_NAME = "base"
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Idiomas (formato NLLB)
SOURCE_LANG = "por_Latn"  # Português
TARGET_LANG = "eng_Latn"  # Inglês

# --- 3. Detecção de Hardware ---
print("Verificando a disponibilidade de aceleradores de hardware (GPU)...")

DEVICE = "cpu"  # Padrão: CPU
message = "Nenhum acelerador de hardware encontrado. Usando CPU (pode ser mais lento)."

# Ordem de preferência: CUDA > MPS > OpenVINO > XPU
if torch.cuda.is_available():  # GPUs NVIDIA/AMD
    DEVICE = "cuda"
    message = "GPU NVIDIA (CUDA) ou AMD (ROCm) encontrada! Usando para aceleração."
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # Apple Silicon
    DEVICE = "mps"
    message = "GPU Apple Silicon (MPS) encontrada! Usando para aceleração."
else:
    try:
        import openvino as ov
        DEVICE = "openvino"
        message = "Intel OpenVINO detectado! Usando para aceleração em CPU/iGPU."
    except ImportError:
        pass # Se OpenVINO não estiver instalado, continua a verificação

    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            DEVICE = "xpu"
            message = "GPU Intel (XPU) encontrada! Usando para aceleração.\nAVISO: O modelo Whisper pode não ser totalmente compatível com XPU nativamente."
    except ImportError:
        pass

print(message)

# --- 4. Carregamento dos Modelos ---
print(f"\nCarregando modelo de transcrição Whisper ({WHISPER_MODEL_NAME}) via faster-whisper...")

# Parâmetros de carregamento do Whisper com base no hardware
whisper_device = "openvino" if DEVICE == "openvino" else DEVICE
compute_type = "int8" if DEVICE in ["openvino", "cpu"] else "float16" if DEVICE == "cuda" else "default"
model_kwargs = {}
if whisper_device == "openvino":
    model_kwargs["openvino_device"] = "GPU"

try:
    print(f"Tentando carregar com device='{whisper_device}', compute_type='{compute_type}', kwargs={model_kwargs}...")
    whisper_model = WhisperModel(WHISPER_MODEL_NAME, device=whisper_device, compute_type=compute_type, **model_kwargs)
    print("Modelo Whisper carregado com sucesso.")

except Exception as e:
    print(f"\nAVISO: Falha ao carregar o modelo com '{DEVICE}'. Erro: {e}")
    print("Tentando carregar o modelo em modo de compatibilidade (CPU)...")

    # Fallback para CPU
    DEVICE = "cpu"
    whisper_device = "cpu"
    compute_type = "int8"
    
    try:
        whisper_model = WhisperModel(WHISPER_MODEL_NAME, device=whisper_device, compute_type=compute_type)
        print("Modelo Whisper carregado com sucesso em modo de compatibilidade (CPU).")
    except Exception as e_cpu:
        print(f"ERRO CRÍTICO: Falha ao carregar o modelo Whisper até mesmo na CPU. Erro: {e_cpu}")
        print(f"Verifique sua instalação do 'faster-whisper' e se o modelo '{WHISPER_MODEL_NAME}' é válido.")
        exit()

print(f"\nCarregando modelo de tradução ({TRANSLATION_MODEL_NAME})...")
# O modelo de tradução não tem backend "openvino", então usa CPU nesse caso.
translation_device = "cpu" if DEVICE in ["openvino", "cpu"] else DEVICE
translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME, src_lang=SOURCE_LANG)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME).to(translation_device)
print(f"Modelo de tradução carregado com sucesso no dispositivo '{translation_device}'.")

# --- 5. Processamento em Tempo Real ---
# Fila para comunicação entre a captura de áudio e a thread de processamento
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback do sounddevice, chamado para cada novo bloco de áudio."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def process_audio_thread():
    """Thread que processa o áudio da fila."""
    print("\nThread de processamento iniciada. Ouvindo o microfone...")

    # Buffer para acumular o áudio
    audio_buffer = np.array([], dtype=DTYPE)

    while True:
        try:
            # Pega o áudio da fila (bloqueante)
            audio_chunk = audio_queue.get()
            audio_buffer = np.concatenate((audio_buffer, audio_chunk.flatten()))

            # Se temos áudio suficiente no buffer, processamos
            if len(audio_buffer) >= CHUNK_SAMPLES:

                # Pega a quantidade exata de áudio para processar
                process_data = audio_buffer[:CHUNK_SAMPLES]
                # Janela deslizante: remove apenas uma parte do início do buffer
                audio_buffer = audio_buffer[SLIDE_SAMPLES:]

                # --- Transcrição ---
                # Otimizações: VAD (Voice Activity Detection) para ignorar silêncio
                # e beam_size=1 para maior velocidade.
                segments, info = whisper_model.transcribe(
                    process_data, 
                    language='pt', 
                    beam_size=1, 
                    vad_filter=True, 
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                transcribed_text = "".join(segment.text for segment in segments).strip()

                # Ignora transcrições vazias ou muito curtas
                if not transcribed_text or len(transcribed_text) < 3:
                    continue

                print(f"PT > {transcribed_text}")

                # --- Tradução ---
                tokenized_text = translation_tokenizer(transcribed_text, return_tensors="pt").to(translation_device)

                translated_tokens = translation_model.generate(
                    **tokenized_text,
                    forced_bos_token_id=translation_tokenizer.convert_tokens_to_ids(TARGET_LANG)
                )

                translated_text = translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
                print(f"EN > {translated_text}\n" + "-"*40)

        except Exception as e:
            print(f"Ocorreu um erro na thread de processamento: {e}")
            break

# --- 6. Função Principal ---
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("==================================================")
    print("      INICIANDO O TRADUTOR EM TEMPO REAL      ")
    print("==================================================")

    # Cria e inicia a thread de processamento
    processor = threading.Thread(target=process_audio_thread, daemon=True)
    processor.start()

    # Inicia a captura de áudio
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, blocksize=BLOCK_SIZE, callback=audio_callback):
            print("\nO programa está rodando. Fale continuamente.")
            print("Pressione ENTER para parar.")
            input() # Aguarda o Enter do usuário

    except Exception as e:
        print(f"\nOcorreu um erro ao iniciar a captura de áudio: {e}")
    finally:
        print("\nEncerrando o programa... Até mais!")

# --- Ponto de Entrada ---
if __name__ == "__main__":
    main()