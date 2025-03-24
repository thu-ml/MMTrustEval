import base64
import io
import os
import time
from typing import List

from PIL import Image
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models

from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry


@registry.register_chatmodel()
class HunyuanChat(BaseChat):
    """
    Chat class for Hunyuan model from Tencent
    """

    model_family = ["hunyuan-vision"]

    model_arch = "hunyuan"

    def __init__(self, model_id: str = "hunyuan-vision", **kargs):
        super().__init__(model_id=model_id)
        # 实例化一个认证对象，入参需要传入腾讯云账户secretId，secretKey
        self.cred = credential.Credential(
            os.getenv("TENCENTCLOUD_SECRET_ID", ""),
            os.getenv("TENCENTCLOUD_SECRET_KEY", ""),
        )

        self.cpf = ClientProfile()
        # 预先建立连接可以降低访问延迟
        self.cpf.httpProfile.pre_conn_pool_size = 3
        self.client = hunyuan_client.HunyuanClient(self.cred, "ap-guangzhou", self.cpf)

        act_req = models.ActivateServiceRequest()
        resp = self.client.ActivateService(act_req)
        self.max_retries = 10
        self.timeout = 1

    def chat(self, messages: List, **generation_kwargs):
        conversation = []
        assert len(messages) == 1, "Only support one-turn conversation currently"
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:

                msg = models.Message()

                if isinstance(message["content"], dict):

                    # multimodal content
                    image_path = message["content"]["image_path"]
                    msg.Contents = [models.Content(), models.Content()]
                    msg.Contents[1].Type = "image_url"
                    imgurl = models.ImageUrl()
                    imgurl.Url = (
                        f"data:image/jpeg;base64,{self.encode_image(image_path)}"
                        if os.path.exists(image_path)
                        else image_path
                    )
                    msg.Contents[1].ImageUrl = imgurl
                    msg.Contents[0].Type = "text"
                    msg.Contents[0].Text = message["content"]["text"]

                else:
                    msg.Content = message["content"]

                msg.Role = message["role"]
                conversation.append(msg)
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )

        req = models.ChatCompletionsRequest()
        req.Messages = conversation
        req.Model = self.model_id
        req.Stream = False

        do_sample = generation_kwargs.get("do_sample", False)
        req.Temperature = (
            generation_kwargs.get("temperature", 1.0) if do_sample else 0.0
        )
        from pprint import pp

        pp(req)

        for i in range(self.max_retries):
            try:
                response = self.client.ChatCompletions(req)
                break
            except TencentCloudSDKException as err:
                print(f"Error in generation: {err}")
                response = f"Error in generation: {err}"
                time.sleep(self.timeout)
        if isinstance(response, str):
            return Response(self.model_id, response, None, None)

        response_message = response.Choices[0].Message.Content
        finish_reason = response.Choices[0].FinishReason
        logprobs = None

        return Response(self.model_id, response_message, logprobs, finish_reason)

    # Function to encode the image
    @classmethod
    def encode_image(cls, image_path: str):
        buffer = io.BytesIO()
        with open(image_path, "rb") as image_file:
            img_data = base64.b64encode(image_file.read())

            img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("RGB")
            print(img.size)
            if img.width > 400 or img.height > 400:
                if img.width > img.height:
                    new_width = 400
                    concat = float(new_width / float(img.width))
                    size = int((float(img.height) * float(concat)))
                    img = img.resize((new_width, size), Image.LANCZOS)
                else:
                    new_height = 400
                    concat = float(new_height / float(img.height))
                    size = int((float(img.width) * float(concat)))
                    img = img.resize((size, new_height), Image.LANCZOS)
                img.save(buffer, format="JPEG")
                img_data = base64.b64encode(buffer.getvalue())
            return img_data.decode("utf-8")
