# capture_mic.py - Módulo para Captura de Áudio do Microfone
import sounddevice as sd
import threading

class MicCapture:
    """
    Gerencia a captura de áudio do microfone em uma thread separada
    para não bloquear a aplicação principal.
    """
    def __init__(self, audio_queue, sample_rate, channels, dtype, block_size):
        """
        Inicializa o capturador de áudio.

        Args:
            audio_queue (queue.Queue): Fila para colocar os blocos de áudio capturados.
            sample_rate (int): Taxa de amostragem.
            channels (int): Número de canais.
            dtype (str): Tipo de dado (ex: 'float32').
            block_size (int): Tamanho do bloco de áudio.
        """
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.block_size = block_size
        
        self.stream = None
        self._running = threading.Event()

    def _audio_callback(self, indata, frames, time, status):
        """Callback chamado pelo sounddevice com novos dados de áudio."""
        if self._running.is_set():
            self.audio_queue.put(indata.copy())

    def start(self):
        """Inicia a captura de áudio."""
        if self.stream and self.stream.active:
            return

        self._running.set()
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, channels=self.channels,
            dtype=self.dtype, blocksize=self.block_size, callback=self._audio_callback
        )
        self.stream.start()

    def stop(self):
        """Para a captura de áudio."""
        self._running.clear()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None