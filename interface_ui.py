# interface_ui.py - Interface Gráfica para o Tradutor de Áudio
import sys
import queue
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel)
from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, Qt, QPoint
from PyQt6.QtGui import QMouseEvent, QFont
from tradutor import Tradutor
from capture_mic import MicCapture

# --- 1. Configurações ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'float32'
BLOCK_SIZE = 1024
CHUNK_SECONDS = 3  # Tamanho do chunk de áudio para processamento (em segundos)
CHUNK_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE)
SLIDE_SECONDS = 1  # Sobreposição da janela deslizante (em segundos)
SLIDE_SAMPLES = int(SLIDE_SECONDS * SAMPLE_RATE)
WHISPER_MODEL_NAME = "base"
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# --- 2. Worker para Processamento em Background ---
class Worker(QObject):
    """
    Executa o carregamento dos modelos e o processamento de áudio em uma
    QThread separada para não bloquear a interface gráfica.
    """
    # Sinais para comunicação com a thread principal (GUI)
    finished = pyqtSignal()
    status_updated = pyqtSignal(str)
    transcribed_text_updated = pyqtSignal(str)
    translated_text_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    processing_started = pyqtSignal()

    def __init__(self, audio_queue):
        super().__init__()
        self.audio_queue = audio_queue
        self.running = False
        # O processador agora é um membro do worker
        self.processor = Tradutor(
            whisper_model_name=WHISPER_MODEL_NAME,
            translation_model_name=TRANSLATION_MODEL_NAME,
            status_callback=self.status_updated.emit,
            error_callback=self.error_occurred.emit
        )

    def stop(self):
        """Sinaliza para a thread parar."""
        self.running = False

    def load_models(self):
        """Delega o carregamento dos modelos para o AudioProcessor."""
        if not self.processor.load_models():
            self.stop()

    @pyqtSlot()
    def process_audio(self):
        """Loop principal de processamento de áudio."""
        self.processing_started.emit()
        self.running = True
        audio_buffer = np.array([], dtype=DTYPE)
        
        while self.running:
            try:
                # Espera por áudio na fila com timeout para poder verificar self.running
                audio_chunk = self.audio_queue.get(timeout=1)
                audio_buffer = np.concatenate((audio_buffer, audio_chunk.flatten()))

                if len(audio_buffer) >= CHUNK_SAMPLES:
                    process_data = audio_buffer[:CHUNK_SAMPLES]
                    audio_buffer = audio_buffer[SLIDE_SAMPLES:]

                    # Delega o processamento para a classe especializada
                    result = self.processor.process_chunk(process_data)

                    if result:
                        transcribed_text, translated_text = result
                        self.transcribed_text_updated.emit(transcribed_text)
                        self.translated_text_updated.emit(translated_text)

            except queue.Empty:
                # Timeout, apenas continua o loop para verificar self.running
                continue
            except Exception as e:
                self.error_occurred.emit(f"Erro no processamento: {e}")
                self.running = False

        self.finished.emit()

# --- 3. Janela Principal da GUI ---
class SubtitleWindow(QMainWindow):
    # Sinal para iniciar o processamento de áudio no worker
    start_processing_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        
        # Variáveis de estado
        self.is_running = False
        self.display_mode = "EN" # "EN", "PT", "SPLIT"
        self.last_pt_text = ""
        self.last_en_text = ""
        self.old_pos = QPoint()
        
        # Thread de processamento
        self.worker_thread = None
        self.worker = None
        self.audio_queue = queue.Queue()

        # Módulo de captura de áudio
        self.mic_capture = MicCapture(self.audio_queue, SAMPLE_RATE, CHANNELS, DTYPE, BLOCK_SIZE)


        # Widget para redimensionamento
        self.grip = None

        # Configuração da UI
        self.init_ui()
        self.setup_worker_thread()

    def init_ui(self):
        """Cria a interface estilo legenda."""
        # Janela sem borda, no topo, e com fundo translúcido.
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(100, 100, 800, 200) # Janela inicial um pouco maior

        container = QWidget()
        container.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
        """)
        self.setCentralWidget(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 0, 10, 5) # Margem superior menor

        # Painel de controle no topo
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        
        btn_style = """ /* Estilo dos botões de controle */
            QPushButton {
                color: white;
                background-color: transparent;
                border: none;
                font-size: 14px;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); }
            QPushButton:pressed { background-color: rgba(255, 255, 255, 0.1); }
        """

        # Botão Iniciar/Parar
        self.start_stop_button = QPushButton("▶")
        self.start_stop_button.setToolTip("Iniciar/Parar Tradução")
        self.start_stop_button.setStyleSheet(btn_style)
        self.start_stop_button.setFixedSize(25, 25)
        self.start_stop_button.setEnabled(False)
        self.start_stop_button.clicked.connect(self.toggle_translation)

        # Botões de visualização e fechar
        self.lang_button = QPushButton("PT/EN")
        self.lang_button.setToolTip("Alternar Idioma (PT/EN)")
        self.lang_button.setStyleSheet(btn_style)
        self.lang_button.setFixedSize(50, 25)
        self.lang_button.clicked.connect(self.toggle_language)

        self.split_button = QPushButton("||")
        self.split_button.setToolTip("Dividir Visão")
        self.split_button.setStyleSheet(btn_style)
        self.split_button.setFixedSize(25, 25)
        self.split_button.clicked.connect(self.toggle_split_view)

        self.close_button = QPushButton("✕")
        self.close_button.setToolTip("Fechar")
        self.close_button.setStyleSheet(btn_style)
        self.close_button.setFixedSize(25, 25)
        self.close_button.clicked.connect(self.close)

        control_layout.addStretch() # Empurra os botões para a direita
        control_layout.addWidget(self.start_stop_button)
        control_layout.addWidget(self.lang_button)
        control_layout.addWidget(self.split_button)
        control_layout.addWidget(self.close_button)

        # Label principal para exibir a legenda
        self.subtitle_label = QLabel("Aguardando início...")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet("color: white; background-color: transparent;")
        
        # Adiciona os layouts e widgets à janela
        layout.addLayout(control_layout)
        layout.addWidget(self.subtitle_label, 1) # Ocupa o espaço restante

        # --- "Pegador" para redimensionar a janela ---
        self.grip = QWidget(self)
        self.grip.setFixedSize(15, 15)
        self.grip.setStyleSheet("background-color: transparent;")
        self.grip.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.grip.mousePressEvent = self.grip_mouse_press
        self.grip.mouseMoveEvent = self.grip_mouse_move

    def resizeEvent(self, event):
        """Mantém o 'pegador' de redimensionamento no canto inferior direito."""
        super().resizeEvent(event)
        if self.grip:
            self.grip.move(self.width() - self.grip.width(), self.height() - self.grip.height())

    def grip_mouse_press(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def grip_mouse_move(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.resize(self.width() + delta.x(), self.height() + delta.y())
            self.old_pos = event.globalPosition().toPoint()

    def setup_worker_thread(self):
        """Cria e configura a thread do worker."""
        self.worker_thread = QThread()
        self.worker = Worker(self.audio_queue)
        self.worker.moveToThread(self.worker_thread)

        # Conecta os sinais do worker aos slots da GUI
        self.worker.status_updated.connect(self.update_subtitle_text)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.transcribed_text_updated.connect(self.update_transcribed_text)
        self.worker.translated_text_updated.connect(self.update_translated_text)
        self.worker.processing_started.connect(lambda: self.update_subtitle_text("Ouvindo..."))

        # Conecta o sinal para iniciar o processamento
        self.start_processing_signal.connect(self.worker.process_audio)

        # Ao iniciar a thread, a primeira tarefa é carregar os modelos
        self.worker_thread.started.connect(self.worker.load_models)
        self.worker.status_updated.connect(lambda msg: self.start_stop_button.setEnabled("Pronto para iniciar" in msg))
        self.worker_thread.start()

    def toggle_translation(self):
        if not self.is_running:
            self.start_translation()
        else:
            self.stop_translation()

    def start_translation(self):
        try:
            self.mic_capture.start()

            # Inicia o processamento na thread do worker
            self.start_processing_signal.emit()
            
            self.is_running = True
            self.start_stop_button.setText("■")
        except Exception as e:
            self.show_error(f"Erro ao iniciar captura de áudio: {e}")

    def stop_translation(self):
        """Para a captura e o processamento de áudio."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.start_stop_button.setText("▶")
        self.update_subtitle_text("Tradução parada.")
        self.mic_capture.stop()

        # Para o loop do worker e limpa a fila de áudio
        if self.worker:
            self.worker.stop()
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()

    def toggle_language(self):
        if self.display_mode == "EN": self.display_mode = "PT"
        else: self.display_mode = "EN"
        self.update_display()

    def toggle_split_view(self):
        if self.display_mode == "SPLIT": self.display_mode = "EN"
        else: self.display_mode = "SPLIT"
        self.update_display()

    def update_display(self):
        """Atualiza o texto da legenda com base no modo de exibição atual."""
        if self.display_mode == "PT":
            self.subtitle_label.setText(self.last_pt_text or "...")
        elif self.display_mode == "EN":
            self.subtitle_label.setText(self.last_en_text or "...")
        elif self.display_mode == "SPLIT":
            self.subtitle_label.setText(f"PT: {self.last_pt_text}\nEN: {self.last_en_text}")

    @pyqtSlot(str)
    def update_subtitle_text(self, text):
        self.subtitle_label.setText(text)

    @pyqtSlot(str)
    def update_transcribed_text(self, text):
        self.last_pt_text = text
        self.update_display()

    @pyqtSlot(str)
    def update_translated_text(self, text):
        self.last_en_text = text
        self.update_display()

    @pyqtSlot(str)
    def show_error(self, error_message):
        self.subtitle_label.setText(f"ERRO: {error_message}")
        print(f"ERRO: {error_message}")
        self.stop_translation()

    # Métodos para mover a janela (arrastando com o mouse)
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton and self.old_pos:
            delta = QPoint(event.globalPosition().toPoint() - self.old_pos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPosition().toPoint()

    def closeEvent(self, event):
        """Garante que a thread do worker seja encerrada ao fechar a janela."""
        self.stop_translation()
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = SubtitleWindow()
    main_window.show()
    sys.exit(app.exec())