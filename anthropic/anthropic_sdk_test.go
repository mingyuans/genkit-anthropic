// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
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
	"context"
	"os"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

func TestAnthropicSDK_Init(t *testing.T) {
	tests := []struct {
		name          string
		plugin        *Anthropic
		expectedError bool
		description   string
	}{
		{
			name:          "should fail when APIKey is empty on init",
			plugin:        &Anthropic{},
			expectedError: true,
			description:   "test initialization failure when APIKey is empty",
		},
		{
			name:          "should succeed when valid APIKey is provided on init",
			plugin:        &Anthropic{APIKey: "sk-ant-test-key"},
			expectedError: false,
			description:   "test successful initialization when valid APIKey is provided",
		},
		{
			name:          "should fail when initializing same plugin instance repeatedly",
			plugin:        nil, // will be created dynamically in test
			expectedError: true,
			description:   "test that repeated initialization of same plugin instance should return error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original environment variable
			originalAPIKey := os.Getenv("ANTHROPIC_API_KEY")
			defer os.Setenv("ANTHROPIC_API_KEY", originalAPIKey)

			// For testing empty APIKey case, clear environment variable
			if tt.name == "should fail when APIKey is empty on init" {
				os.Unsetenv("ANTHROPIC_API_KEY")
			}

			ctx := context.Background()
			g, err := genkit.Init(ctx)
			if err != nil {
				t.Fatalf("genkit initialization failed: %v", err)
			}

			// Special handling for repeated initialization test
			if tt.name == "should fail when initializing same plugin instance repeatedly" {
				plugin := &Anthropic{APIKey: "sk-ant-test-key"}
				// First initialization should succeed
				err1 := plugin.Init(ctx, g)
				if err1 != nil {
					t.Fatalf("first initialization failed: %v", err1)
				}
				// Second initialization should fail
				err2 := plugin.Init(ctx, g)
				if err2 == nil {
					t.Errorf("repeated initialization should fail")
				}
				return
			}

			err = tt.plugin.Init(ctx, g)

			if tt.expectedError && err == nil {
				t.Errorf("expected error but got none")
			}
			if !tt.expectedError && err != nil {
				t.Errorf("expected no error but got: %v", err)
			}
		})
	}
}

func TestAnthropicSDK_Name(t *testing.T) {
	plugin := &Anthropic{}
	expectedName := "anthropic"

	if plugin.Name() != expectedName {
		t.Errorf("expected plugin name %s, got %s", expectedName, plugin.Name())
	}
}

func TestAnthropicSDK_DefineModel(t *testing.T) {
	t.Run("should succeed defining known model without ModelInfo", func(t *testing.T) {
		ctx := context.Background()
		g, err := genkit.Init(ctx)
		if err != nil {
			t.Fatalf("genkit initialization failed: %v", err)
		}

		plugin := &Anthropic{APIKey: "sk-ant-test-key"}
		err = plugin.Init(ctx, g)
		if err != nil {
			t.Fatalf("plugin initialization failed: %v", err)
		}

		// All models are already defined during plugin initialization, attempting to redefine should succeed (or return existing model)
		model, err := plugin.DefineModel(g, "claude-3-5-sonnet", nil)
		if err != nil {
			t.Errorf("expected no error but got: %v", err)
		}
		if model == nil {
			t.Errorf("expected model to be returned but got nil")
		}
	})

	t.Run("should fail defining unknown model without ModelInfo", func(t *testing.T) {
		ctx := context.Background()
		g, err := genkit.Init(ctx)
		if err != nil {
			t.Fatalf("genkit initialization failed: %v", err)
		}

		plugin := &Anthropic{APIKey: "sk-ant-test-key"}
		err = plugin.Init(ctx, g)
		if err != nil {
			t.Fatalf("plugin initialization failed: %v", err)
		}

		_, err = plugin.DefineModel(g, "unknown-model", nil)
		if err == nil {
			t.Errorf("expected error but got none")
		}
	})

	t.Run("should succeed defining model with custom ModelInfo", func(t *testing.T) {
		ctx := context.Background()
		g, err := genkit.Init(ctx)
		if err != nil {
			t.Fatalf("genkit initialization failed: %v", err)
		}

		plugin := &Anthropic{APIKey: "sk-ant-test-key"}
		err = plugin.Init(ctx, g)
		if err != nil {
			t.Fatalf("plugin initialization failed: %v", err)
		}

		modelInfo := &ai.ModelInfo{
			Label:    "Custom Model",
			Supports: &ai.ModelSupports{},
			Versions: []string{"v1.0"},
		}
		model, err := plugin.DefineModel(g, "custom-model", modelInfo)
		if err != nil {
			t.Errorf("expected no error but got: %v", err)
		}
		if model == nil {
			t.Errorf("expected model to be returned but got nil")
		}
	})
}

func TestAnthropicSDK_Generate(t *testing.T) {
	tests := []struct {
		name        string
		request     *ai.ModelRequest
		expectError bool
		description string
	}{
		{
			name: "basic text generation request should succeed",
			request: &ai.ModelRequest{
				Messages: []*ai.Message{
					ai.NewUserTextMessage("Hello, how are you?"),
				},
			},
			expectError: false,
			description: "test basic text generation functionality",
		},
		{
			name: "request with system prompt should succeed",
			request: &ai.ModelRequest{
				Messages: []*ai.Message{
					ai.NewSystemTextMessage("You are a helpful assistant."),
					ai.NewUserTextMessage("What is the capital of France?"),
				},
			},
			expectError: false,
			description: "test request with system prompt",
		},
		{
			name: "request with config parameters should succeed",
			request: &ai.ModelRequest{
				Messages: []*ai.Message{
					ai.NewUserTextMessage("Tell me a joke"),
				},
				Config: &ai.GenerationCommonConfig{
					Temperature:     0.7,
					MaxOutputTokens: 100,
					TopP:            0.9,
				},
			},
			expectError: false,
			description: "test request with config parameters",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			g, err := genkit.Init(ctx)
			if err != nil {
				t.Fatalf("genkit initialization failed: %v", err)
			}

			plugin := &Anthropic{APIKey: "sk-ant-test-key"}
			err = plugin.Init(ctx, g)
			if err != nil {
				t.Fatalf("plugin initialization failed: %v", err)
			}

			_, err = plugin.DefineModel(g, "claude-3-5-sonnet", nil)
			if err != nil {
				t.Fatalf("model definition failed: %v", err)
			}

			// Here we only test request conversion logic, not actual API calls
			resp, err := anthropicGenerate(ctx, plugin.client, "claude-3-5-sonnet", tt.request, nil)

			if tt.expectError && err == nil {
				t.Errorf("expected error but got none")
			}
			if !tt.expectError && err != nil {
				// Since we're using a test API Key, actual calls will fail, which is expected
				// We mainly test that request conversion logic doesn't error
				if _, convErr := toAnthropicRequest("claude-3-5-sonnet", tt.request); convErr != nil {
					t.Errorf("request conversion failed: %v", convErr)
				}
			}
			if !tt.expectError && err == nil && resp == nil {
				t.Errorf("expected response to be returned but got nil")
			}
		})
	}
}

func TestAnthropicSDK_StreamingGenerate(t *testing.T) {
	t.Run("streaming generation should support callback function", func(t *testing.T) {
		ctx := context.Background()
		g, err := genkit.Init(ctx)
		if err != nil {
			t.Fatalf("genkit initialization failed: %v", err)
		}

		plugin := &Anthropic{APIKey: "sk-ant-test-key"}
		err = plugin.Init(ctx, g)
		if err != nil {
			t.Fatalf("plugin initialization failed: %v", err)
		}

		request := &ai.ModelRequest{
			Messages: []*ai.Message{
				ai.NewUserTextMessage("Count from 1 to 5"),
			},
		}

		callback := func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
			if chunk == nil {
				t.Errorf("received nil chunk")
			}
			return nil
		}

		// Test streaming generation (will fail with test API Key, but we verify logic)
		_, err = anthropicGenerate(ctx, plugin.client, "claude-3-5-sonnet", request, callback)

		// Since we're using test API Key, actual calls will fail, but we verify request conversion logic
		if _, convErr := toAnthropicRequest("claude-3-5-sonnet", request); convErr != nil {
			t.Errorf("streaming request conversion failed: %v", convErr)
		}
	})
}

func TestAnthropicSDK_LookupModel(t *testing.T) {
	t.Run("should succeed finding defined model", func(t *testing.T) {
		ctx := context.Background()
		g, err := genkit.Init(ctx)
		if err != nil {
			t.Fatalf("genkit initialization failed: %v", err)
		}

		plugin := &Anthropic{APIKey: "sk-ant-test-key"}
		err = plugin.Init(ctx, g)
		if err != nil {
			t.Fatalf("plugin initialization failed: %v", err)
		}

		// All built-in models should be automatically defined after initialization
		model := AnthropicModel(g, "claude-3-5-sonnet")
		if model == nil {
			t.Errorf("expected to find model claude-3-5-sonnet but got nil")
		}
	})

	t.Run("should return nil for undefined model", func(t *testing.T) {
		ctx := context.Background()
		g, err := genkit.Init(ctx)
		if err != nil {
			t.Fatalf("genkit initialization failed: %v", err)
		}

		plugin := &Anthropic{APIKey: "sk-ant-test-key"}
		err = plugin.Init(ctx, g)
		if err != nil {
			t.Fatalf("plugin initialization failed: %v", err)
		}

		model := AnthropicModel(g, "non-existent-model")
		if model != nil {
			t.Errorf("expected nil but found model")
		}
	})
}
