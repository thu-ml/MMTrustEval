from typing import List, Optional
from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry
from PIL import Image

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers.models.instructblip.modeling_instructblip import logger
import torch


class InstructBlipForConditionalGenerationV2(InstructBlipForConditionalGeneration):
    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            qformer_input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt to be fed to the Q-Former module.
            qformer_attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices.
            interpolate_pos_encoding (`bool`, *optional*, defaults to `False`):
                Whether to interpolate the positional encoding of the image embeddings.

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        if pixel_values is not None:
            batch_size = pixel_values.shape[0]
            image_embeds = self.vision_model(
                pixel_values,
                return_dict=True,
                interpolate_pos_encoding=interpolate_pos_encoding,
            ).last_hidden_state
        else:
            batch_size = 1

        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(self.vision_model.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1],
                dtype=torch.long,
                device=self.vision_model.device,
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_attention_mask = torch.ones(
                query_tokens.size()[:-1],
                dtype=torch.long,
                device=self.vision_model.device,
            )
            if qformer_attention_mask is None:
                qformer_attention_mask = torch.ones_like(qformer_input_ids)
            qformer_attention_mask = torch.cat(
                [query_attention_mask, qformer_attention_mask], dim=1
            )
            query_outputs = self.qformer(
                input_ids=qformer_input_ids,
                attention_mask=qformer_attention_mask,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]

            language_model_inputs = self.language_projection(query_output)
            language_attention_mask = torch.ones(
                language_model_inputs.size()[:-1],
                dtype=torch.long,
                device=language_model_inputs.device,
            )

            # if the model already has "image_token_index" then the input is expanded to account for image embeds
            # otherwise we expand manually by concatenating
            if getattr(self.config, "image_token_index", None) is not None:
                special_image_mask = (
                    (input_ids == self.config.image_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                inputs_embeds[special_image_mask] = language_model_inputs.flatten()
            else:
                logger.warning_once(
                    "Expanding inputs for image tokens in InstructBLIP should be done in processing. "
                    "Please follow instruction here (https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042) to update your InstructBLIP model. "
                    "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                )
                inputs_embeds = torch.cat(
                    [
                        language_model_inputs,
                        inputs_embeds.to(language_model_inputs.device),
                    ],
                    dim=1,
                )
                attention_mask = torch.cat(
                    [
                        language_attention_mask,
                        attention_mask.to(language_attention_mask.device),
                    ],
                    dim=1,
                )

                # add image_embeds length to max_length, so that the final max_length in counted only on token embeds
                # -1 is to account for the prepended BOS after `generate.`
                if not self.language_model.config.is_encoder_decoder:
                    generate_kwargs["max_length"] = (
                        generate_kwargs.get("max_length", 20)
                        + language_model_inputs.shape[1]
                        - 1
                    )
                    generate_kwargs["min_length"] = (
                        generate_kwargs.get("min_length", 0)
                        + language_model_inputs.shape[1]
                    )

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # this is a temporary workaround to be consistent with other generation models and
        # have BOS as the first token, even though under the hood we are calling LM with embeds
        if not self.language_model.config.is_encoder_decoder:
            # the InstructBLIP authors used inconsistent tokenizer/model files during training,
            # with the tokenizer's bos token being set to </s> which has ID=2,
            # whereas the model's text config has bos token id = 0
            bos_token_id = (
                2
                if self.config.text_config.architectures[0] == "LLaMAForCausalLM"
                else self.config.text_config.bos_token_id
            )
            bos_tokens = (
                torch.LongTensor([[bos_token_id]])
                .repeat(batch_size, 1)
                .to(self.vision_model.device)
            )
            if not isinstance(outputs, torch.Tensor):
                outputs.sequences = torch.cat([bos_tokens, outputs.sequences], dim=-1)
            else:
                outputs = torch.cat([bos_tokens, outputs], dim=-1)

        return outputs


@registry.register_chatmodel()
class InstructBLIPChat(BaseChat):
    """
    Chat class for INSTRUCTBLIP models
    """

    MODEL_CONFIG = {
        "instructblip-flan-t5-xxl": "Salesforce/instructblip-flan-t5-xxl",
    }

    model_family = list(MODEL_CONFIG.keys())

    model_arch = "instructblip"

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        model_path = self.MODEL_CONFIG[self.model_id]
        self.device = device
        self.model = InstructBlipForConditionalGenerationV2.from_pretrained(
            model_path
        ).to(self.device)
        self.processor = InstructBlipProcessor.from_pretrained(model_path)

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        # TODO: if system message provided.
        assert len(messages) == 1, "Only support one-turn conversation currently"
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        raw_image = Image.open(image_path).convert("RGB")
                        inputs = self.processor(
                            images=raw_image, text=user_message, return_tensors="pt"
                        ).to(self.device)
                    else:
                        user_message = message["content"]
                        inputs = self.processor(
                            text=user_message, return_tensors="pt"
                        ).to(self.device)

                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )

            generation_config = {
                "do_sample": False,
                "max_new_tokens": 512,
                "temperature": 1,
                "num_beams": 5,
                "min_length": 1,
                "repetition_penalty": 1.5,
                "length_penalty": 1.0,
            }
            generation_config.update(generation_kwargs)
            if not generation_config["do_sample"]:
                generation_config["temperature"] = 0

            from pprint import pp

            pp(generation_config)

            outputs = self.model.generate(
                **inputs,
                **generation_config,
            )

            generated_text = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0].strip()
        scores = None

        return Response(self.model_id, generated_text, scores, None)
