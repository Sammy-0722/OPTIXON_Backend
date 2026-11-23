using Microsoft.AspNetCore.Http;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace AIManagement.Api.DTO
{
    public class RegisterFaceRequest
    {
        [Required]
        public string Name { get; set; } = string.Empty;
        [Required]
        public string EmployeeId { get; set; } = string.Empty;

        [Required]
        public List<IFormFile> Images { get; set; } = new();
    }
}