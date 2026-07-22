# Registry for the conversion job image. Batch pulls from here.

resource "aws_ecr_repository" "convert" {
  name = "dronesynth-convert"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# cap storage: old untagged layers from repeated pushes get expired
resource "aws_ecr_lifecycle_policy" "convert" {
  repository = aws_ecr_repository.convert.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "keep the last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      }
    ]
  })
}
