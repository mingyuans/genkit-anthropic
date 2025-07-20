package main

import (
	"context"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/mingyuans/genkit-anthropic/anthropic"
	"log"
)

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Anthropic plugin and Claude model.
	g, err := genkit.Init(ctx,
		genkit.WithPlugins(&anthropic.Anthropic{}),
		genkit.WithDefaultModel("anthropic/claude-sonnet-4"),
	)
	if err != nil {
		log.Fatalf("could not initialize Genkit: %v", err)
	}
	resp, err := genkit.Generate(ctx, g, ai.WithPrompt("What is the meaning of life?"))
	if err != nil {
		log.Fatalf("could not generate model response: %v", err)
	}

	log.Println(resp.Text())
}
