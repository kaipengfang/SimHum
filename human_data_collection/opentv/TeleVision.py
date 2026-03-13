"""
Copyright [2024] [Xuxin Cheng, Jialong Li]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import time
from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, DefaultScene, AmbientLight
from multiprocessing import Value, Array, Process, shared_memory
import numpy as np
import asyncio
from pathlib import Path
import logging
import aiohttp_cors

from .Preprocessor import VuerPreprocessor

class OpenTeleVision:
    def __init__(self,
                 img_shape,
                 shm_name,
                 stream_mode="image",
                 cert_file="./cert.pem",
                 key_file="./key.pem",
                 ngrok=False):
        self.img_shape = (img_shape[0], 2*img_shape[1], 3)
        self.img_height, self.img_width = img_shape[:2]
        self.aspect_ratio = self.img_width / self.img_height

        if ngrok:
            cert_file = None
            key_file = None
        self.app = Vuer(host='0.0.0.0', static_root=Path(__file__).parent / "./assets", cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=100, free_port=True)

        cors = aiohttp_cors.setup(self.app.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })

        # Set up file logging
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Set up console logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only WARNING and above go to console
        console_handler.setFormatter(formatter)

        # Configure root logger
        logging.basicConfig(level=logging.INFO, handlers=[console_handler])

        self.logger = logging.getLogger(__name__)

        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if stream_mode == "image":
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
            self.app.spawn(start=False)(self.main_image)
        else:
            raise ValueError("stream_mode must be 'image'")

        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)

        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

        self.cam_pos = self.head_matrix.copy()[:3, -1]
        self.last_cam_pos = self.head_matrix.copy()[:3, -1]

        self.process = Process(target=self.run)
        self.process.daemon = True
        self.process.start()

        self.processor = VuerPreprocessor()

    def run(self):
        self.app.run()

    def step(self):
        processed = self.processor.process(self)
        return processed

    async def on_cam_move(self, event, session, fps=60):
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        try:
            left_mat = np.array(event.value["left"]).reshape((25, 4, 4))
            right_mat = np.array(event.value["right"]).reshape((25, 4, 4))
            self.left_hand_shared[:] = left_mat[0].flatten()
            self.right_hand_shared[:] = right_mat[0].flatten()
            self.left_landmarks_shared[:] = left_mat[:, 3, :3].flatten()
            self.right_landmarks_shared[:] = right_mat[:, 3, :3].flatten()
        except:
            pass

    async def main_image(self, session, fps=60):
        session.upsert @ Hands(fps=fps, stream=True, show_helper=False, key="hands")
        session.upsert @ AmbientLight(intensity=0.5, show_helper=False, key="ambient")  # avoid helper
        while True:
            start = time.time()
            display_image = self.img_array

            session.upsert(
                [ImageBackground(
                    display_image[:, :self.img_width],
                    format="jpeg",
                    quality=80,
                    key="left-image",
                    interpolate=True,
                    aspect=self.aspect_ratio,
                    height = 8,
                    position=[0, -1, 3],
                    layers=1,
                    alphaSrc="./vinette.jpg"
                ),
                ImageBackground(
                    display_image[:, self.img_width:],
                    format="jpeg",
                    quality=80,
                    key="right-image",
                    interpolate=True,
                    aspect=self.aspect_ratio,
                    height = 8,
                    position=[0, -1, 3],
                    layers=2,
                    alphaSrc="./vinette.jpg"
                )],
                to="bgChildren",
            )
            await asyncio.sleep(0.033)

    @property
    def left_hand(self):
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")


    @property
    def right_hand(self):
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")


    @property
    def left_landmarks(self):
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)

    @property
    def right_landmarks(self):
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        return float(self.aspect_shared.value)
