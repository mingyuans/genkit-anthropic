# Usage

```go
// Initialize Genkit with the Anthropic plugin and Claude model.
g, err := genkit.Init(ctx,
    genkit.WithPlugins(&anthropic.Anthropic{}),
    genkit.WithDefaultModel("anthropic/claude-sonnet-4"),
)
```