"""
Desktop TCP WiFi Bridge Server.
Listens for incoming audio streams from the standalone ESP8266 hardware table node,
feeds raw PCM audio into the DSP/VAD/NLP pipeline, and streams real-time
classification feedback back to the hardware to drive its RGB LED and OLED display.
"""

import socket
import struct
import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple

class WiFiHardwareBridge:
    """
    TCP Server bridging the physical ESP8266 table node with the Python DSP engine.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5555):
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.client_address: Optional[Tuple[str, int]] = None
        self.is_running: bool = False
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    def start_server(self):
        """Start listening for ESP8266 connection in background thread."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.is_running = True

        print(f"[WiFi Bridge] Server listening on {self.host}:{self.port}")
        self._thread = threading.Thread(target=self._server_loop, daemon=True)
        self._thread.start()

    def _server_loop(self):
        """Accept connection and stream audio."""
        while self.is_running:
            try:
                print("[WiFi Bridge] Waiting for ESP8266 table node to connect...")
                client, addr = self.server_socket.accept()
                self.client_socket = client
                self.client_address = addr
                print(f"[WiFi Bridge]  ESP8266 Node Connected from {addr[0]}:{addr[1]}")

                chunk_bytes = 1600
                while self.is_running:
                    data = bytearray()
                    while len(data) < chunk_bytes:
                        packet = client.recv(chunk_bytes - len(data))
                        if not packet:
                            raise ConnectionResetError("ESP8266 disconnected.")
                        data.extend(packet)

                    samples = np.frombuffer(data, dtype=np.int16)
                    self.audio_queue.put(samples)

            except Exception as e:
                print(f"[WiFi Bridge] Connection notice: {e}")
                if self.client_socket:
                    try:
                        self.client_socket.close()
                    except Exception:
                        pass
                    self.client_socket = None
                time.sleep(1)

    def get_audio_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Retrieve next audio chunk streamed by ESP8266."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def send_hardware_feedback(self, status: str, prof_pct: int = 80):
        """
        Send feedback string back to ESP8266 to update physical RGB LED and OLED.
        Format: "PROF:85\n" or "DISC:20\n" or "ANALYZING\n"
        """
        if self.client_socket is None:
            return

        try:
            if status == "PROFESSIONAL":
                msg = f"PROF:{prof_pct}\n"
            elif status in ("PERSONAL", "DISCARD"):
                msg = "DISC:0\n"
            else:
                msg = "ANALYZING\n"

            self.client_socket.sendall(msg.encode('utf-8'))
        except Exception:
            pass

    def stop(self):
        """Close socket server."""
        self.is_running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        print("[WiFi Bridge] Server stopped.")

if __name__ == "__main__":
    bridge = WiFiHardwareBridge()
    bridge.start_server()
    print("WiFi Bridge running. Press Ctrl+C to stop.")
    try:
        while True:
            chunk = bridge.get_audio_chunk(timeout=1.0)
            if chunk is not None:
                rms = float(np.sqrt(np.mean((chunk / 32768.0) ** 2)))
                print(f"Received ESP8266 Audio Chunk: {len(chunk)} samples (RMS: {rms:.4f})")
    except KeyboardInterrupt:
        bridge.stop()
