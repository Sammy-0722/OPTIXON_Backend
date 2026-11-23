namespace AIManagement.Api.Services;

public sealed record FaceAnnotation(
    int X,
    int Y,
    int Width,
    int Height,
    string Name,
    float Similarity);