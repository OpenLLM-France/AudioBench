#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, April 19th 2024, 11:17:41 am
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
#
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import logging
import pathlib

import google.generativeai as genai

from model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


def _do_sample_inference(self, audio_array, instruction, sampling_rate=16000):

    audio_path = self._write_temp_audio(audio_array, sampling_rate)

    response = self.model.generate_content([
        instruction,
        {
            "mime_type": "audio/wav",
            "data": pathlib.Path(audio_path).read_bytes()
        }
    ])

    response = response.text
    return response


class Gemini2Flash(BaseModel):

    def load(self):
        # Initialize a Gemini model appropriate for your use case.
        self.model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        logger.info("Model loaded")

    def _generate(self, input):

        instruction   = input["instruction"]

        segments, sampling_rate, mode = self._prepare_audio_segments(input["audio"], input['task_type'])

        if mode == 'chunked':
            return ' '.join(_do_sample_inference(self, seg, instruction) for seg in segments)
        return _do_sample_inference(self, segments[0], instruction)
