namespace AIManagement.Api.Services;

public sealed record FaceInferenceResult(
    int X,
    int Y,
    int Width,
    int Height,
    float DetectionConfidence,
    float[] Embedding);