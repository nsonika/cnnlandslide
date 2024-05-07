// document.addEventListener("DOMContentLoaded", function () {
//   var showConfidenceButton = document.getElementById("showConfidence");
//   var confidenceContainer = document.getElementById("confidenceContainer");

//   showConfidenceButton.addEventListener("click", function () {
//     confidenceContainer.style.display =
//       confidenceContainer.style.display === "none" ? "block" : "none";
//   });
// });

document.addEventListener("DOMContentLoaded", function () {
  var showConfidenceButton = document.getElementById("showConfidence");
  var confidenceContainer = document.getElementById("confidenceContainer");

  showConfidenceButton.addEventListener("click", function () {
    // Hide the button and show the confidence container
    showConfidenceButton.style.display = "none"; // Hide the button
    confidenceContainer.style.display = "block"; // Show the confidence
  });
});
