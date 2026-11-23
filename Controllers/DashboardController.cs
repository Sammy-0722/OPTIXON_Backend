using AIManagement.Api.Repositories;
using Microsoft.AspNetCore.Mvc;
using AIManagement.Api.services;

namespace AIManagement.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class DashboardController : ControllerBase
{
    private readonly FaceRepository _faceReposiory;
    

    public DashboardController(FaceRepository faceReposiory)
    {
        _faceReposiory = faceReposiory;
    }
    [HttpGet("summary")]
    public async Task<IActionResult> GetSummary() => Ok(await _faceReposiory.GetDashboardSummaryAsync());
}