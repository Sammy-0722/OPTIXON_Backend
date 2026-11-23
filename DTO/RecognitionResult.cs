namespace AIManagement.Api.DTO
{
    public class RecognitionResult
    {
        public string Name { get; set; } = "Unkown";
        public bool Recognized { get; set; }

        public float? Similarity{ get; set;  }
        public string? EmployeeId { get; set; }

        public float DetectionConfidence { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public float Width { get; set; }
        public float Height { get; set; }
    }
}
