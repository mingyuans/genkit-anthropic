// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"github.com/firebase/genkit/go/ai"
)

const provider = "anthropic"

// Multimodal defines model capabilities for multimodal models
var Multimodal = ai.ModelSupports{
	Multiturn:  true,
	Tools:      true,
	SystemRole: true,
	Media:      true,
}

// supported anthropic models
var anthropicModels = map[string]ai.ModelInfo{
	"claude-3-5-sonnet-v2": {
		Label:    "Anthropic Claude 3.5 Sonnet v2",
		Supports: &Multimodal,
		Versions: []string{"claude-3-5-sonnet-latest"},
	},
	"claude-3-5-sonnet": {
		Label:    "Anthropic Claude 3.5 Sonnet",
		Supports: &Multimodal,
		Versions: []string{"claude-3-5-sonnet-20240620"},
	},
	"claude-3-haiku": {
		Label:    "Anthropic Claude 3 Haiku",
		Supports: &Multimodal,
		Versions: []string{"claude-3-haiku-20240307"},
	},
	"claude-3-5-haiku": {
		Label:    "Anthropic Claude 3.5 Haiku",
		Supports: &Multimodal,
		Versions: []string{"claude-3-5-haiku-latest"},
	},
	"claude-3-7-sonnet": {
		Label:    "Anthropic Claude 3.7 Sonnet",
		Supports: &Multimodal,
		Versions: []string{"claude-3-7-sonnet-latest"},
	},
	"claude-opus-4": {
		Label:    "Anthropic Claude Opus 4",
		Supports: &Multimodal,
		Versions: []string{"claude-opus-4-20250514"},
	},
	"claude-sonnet-4": {
		Label:    "Anthropic Claude Sonnet 4",
		Supports: &Multimodal,
		Versions: []string{"claude-sonnet-4-20250514"},
	},
}
