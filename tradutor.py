# tradutor.py - Módulo de Lógica de Transcrição e Tradução
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("ERRO CRÍTICO: Instale 'faster-whisper' com: pip install faster-whisper")
    raise

# Verifica a dependência opcional para o filtro VAD
try:
    import onnxruntime
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False


class Tradutor:
    """
    Encapsula o carregamento dos modelos e a lógica de processamento de áudio
    (transcrição e tradução), tornando-a reutilizável.
    """

    def __init__(self, whisper_model_name, translation_model_name, status_callback=None, error_callback=None):
        """
        Inicializa o tradutor.

        Args:
            whisper_model_name (str): Nome do modelo Whisper a ser usado.
            translation_model_name (str): Nome do modelo de tradução a ser usado.
            status_callback (callable, optional): Função para reportar o progresso do carregamento.
            error_callback (callable, optional): Função para reportar erros.
        """
        self.whisper_model_name = whisper_model_name
        self.translation_model_name = translation_model_name
        self.status_callback = status_callback or (lambda msg: print(msg))
        self.error_callback = error_callback or (lambda err: print(f"ERRO: {err}"))

        self.whisper_model = None
        self.translation_model = None
        self.translation_tokenizer = None
        self.device = "cpu"

    def _report_status(self, message):
        if self.status_callback:
            self.status_callback(message)

    def _report_error(self, error_message):
        if self.error_callback:
            self.error_callback(error_message)

    def load_models(self):
        """Carrega os modelos de IA, utilizando os callbacks para reportar o status."""
        try:
            self._report_status("Verificando aceleradores de hardware...")

            if torch.cuda.is_available():
                self.device = "cuda"
                compute_type = "float16"
                self._report_status("Usando GPU NVIDIA (CUDA).")
            else:
                self.device = "cpu"
                compute_type = "int8"
                self._report_status("Usando CPU.")

            self._report_status(f"Carregando modelo Whisper '{self.whisper_model_name}'...")
            self.whisper_model = WhisperModel(self.whisper_model_name, device=self.device, compute_type=compute_type)

            self._report_status(f"Carregando modelo de tradução '{self.translation_model_name}'...")
            self.translation_tokenizer = AutoTokenizer.from_pretrained(self.translation_model_name, src_lang="por_Latn")
            self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(self.translation_model_name).to(self.device)

            self._report_status("Modelos carregados. Pronto para iniciar.")
            return True
        except Exception as e:
            self._report_error(f"Erro ao carregar modelos: {e}")
            return False

    def process_chunk(self, audio_chunk):
        """
        Processa um pedaço de áudio, realizando a transcrição e a tradução.

        Args:
            audio_chunk (np.array): O array de áudio a ser processado.

        Returns:
            tuple[str, str] | None: Uma tupla contendo (texto_transcrito, texto_traduzido) ou None se não houver texto.
        """
        if not all([self.whisper_model, self.translation_model, self.translation_tokenizer]):
            self._report_error("Modelos não foram carregados antes do processamento.")
            return None

        # --- Transcrição ---
        segments, _ = self.whisper_model.transcribe(
            audio_chunk, language='pt', beam_size=1, vad_filter=VAD_AVAILABLE,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        transcribed_text = "".join(segment.text for segment in segments).strip()

        if not transcribed_text or len(transcribed_text) < 3:
            return None

        # --- Tradução ---
        tokenized_text = self.translation_tokenizer(transcribed_text, return_tensors="pt").to(self.device)
        translated_tokens = self.translation_model.generate(
            **tokenized_text,
            forced_bos_token_id=self.translation_tokenizer.convert_tokens_to_ids("eng_Latn")
        )
        translated_text = self.translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        return transcribed_text, translated_text