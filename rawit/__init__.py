# Copyright 2025 Wisu Suntoyo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Architectural concepts derived from Laya (Apache-2.0):
# https://github.com/convaiinnovations/laya

"""Rawit — System 1 decision engine for Bahasa Indonesia & Bahasa Melayu."""

from .configuration_rawit import RawitConfig
from .modeling_rawit import RawitModel
from .pipeline import RawitPipeline

__version__ = "0.1.0"
__all__ = ["RawitConfig", "RawitModel", "RawitPipeline"]
