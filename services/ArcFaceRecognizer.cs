using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Runtime.InteropServices;

namespace AIManagement.Api.Services
{
    public class ArcFaceRecognizer : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;

        private const int FaceSize = 112;

        public ArcFaceRecognizer(string modelPath)
        {
            _session = new InferenceSession(modelPath);
            _inputName = _session.InputMetadata.Keys.First();
        }

        public float[] CreateEmbedding(Mat alignedFace)
        {
            using var resized = new Mat();

            Cv2.Resize(
                alignedFace,
                resized,
                new Size(FaceSize, FaceSize)
            );

            using var rgb = new Mat();

            Cv2.CvtColor(
                resized,
                rgb,
                ColorConversionCodes.BGR2RGB
            );

            int pixelCount = FaceSize * FaceSize * 3;

            byte[] pixels = new byte[pixelCount];

            // Bulk copy instead of accessing every pixel with Mat.At()
            Marshal.Copy(
                rgb.Data,
                pixels,
                0,
                pixelCount
            );

            var tensor = new DenseTensor<float>(
                new[] { 1, 3, FaceSize, FaceSize }
            );

            int index = 0;

            for (int y = 0; y < FaceSize; y++)
            {
                for (int x = 0; x < FaceSize; x++)
                {
                    int pixelIndex = (y * FaceSize + x) * 3;

                    float r = (pixels[pixelIndex] - 127.5f) / 127.5f;
                    float g = (pixels[pixelIndex + 1] - 127.5f) / 127.5f;
                    float b = (pixels[pixelIndex + 2] - 127.5f) / 127.5f;

                    tensor[0, 0, y, x] = r;
                    tensor[0, 1, y, x] = g;
                    tensor[0, 2, y, x] = b;
                }
            }
            var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor(_inputName, tensor)
};

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results =
                _session.Run(inputs);

            var output = results.First().AsTensor<float>();

            float[] embedding = output.ToArray();

            // Normalize embedding
            float magnitude = MathF.Sqrt(
                embedding.Sum(value => value * value)
            );

            if (magnitude > 0)
            {
                for (int i = 0; i < embedding.Length; i++)
                {
                    embedding[i] /= magnitude;
                }
            }

            return embedding;
        }

        public void Dispose()
        {
            _session.Dispose();
        }
    }
}