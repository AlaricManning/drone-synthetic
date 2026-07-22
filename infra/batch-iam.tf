# Two roles, two very different jobs:
# - the EXECUTION role is what Fargate itself uses to set the task up
#   (pull the image from ECR, ship logs to CloudWatch) — AWS-managed policy;
# - the JOB role is what our code runs as: read raw runs, write datasets
#   and QC, nothing else. This is the "batch job role" row of the README's
#   security model table.

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_execution" {
  name               = "dronesynth-batch-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "batch_execution" {
  role       = aws_iam_role.batch_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "convert_job" {
  name               = "dronesynth-convert-job"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy" "convert_job" {
  name = "convert-raw-to-datasets"
  role = aws_iam_role.convert_job.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadRawRuns"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.data.arn}/raw/*"
      },
      {
        Sid      = "ListRawRuns"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.data.arn
        Condition = {
          StringLike = { "s3:prefix" = "raw/*" }
        }
      },
      {
        Sid    = "WriteDatasetsAndQc"
        Effect = "Allow"
        Action = ["s3:PutObject"]
        Resource = [
          "${aws_s3_bucket.data.arn}/datasets/*",
          "${aws_s3_bucket.data.arn}/qc/*",
        ]
      }
    ]
  })
}
