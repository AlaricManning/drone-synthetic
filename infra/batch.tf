# AWS Batch on Fargate: submit a run id, a container converts it, nothing
# exists (or costs anything) between jobs.
#
# Networking: jobs run in the default VPC's subnets with a public IP and an
# egress-only security group — outbound to S3/ECR, nothing inbound.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "batch_jobs" {
  name        = "dronesynth-batch-jobs"
  description = "Egress-only for dronesynth conversion jobs"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_batch_compute_environment" "fargate" {
  compute_environment_name = "dronesynth-fargate"
  type                     = "MANAGED"

  compute_resources {
    type               = "FARGATE"
    max_vcpus          = 16
    security_group_ids = [aws_security_group.batch_jobs.id]
    subnets            = data.aws_subnets.default.ids
  }
}

resource "aws_batch_job_queue" "convert" {
  name     = "dronesynth-convert"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.fargate.arn
  }
}

resource "aws_batch_job_definition" "convert" {
  name                  = "dronesynth-convert"
  type                  = "container"
  platform_capabilities = ["FARGATE"]

  container_properties = jsonencode({
    image = "${aws_ecr_repository.convert.repository_url}:latest"
    resourceRequirements = [
      { type = "VCPU", value = "1" },
      { type = "MEMORY", value = "2048" },
    ]
    executionRoleArn = aws_iam_role.batch_execution.arn
    jobRoleArn       = aws_iam_role.convert_job.arn
    networkConfiguration = {
      assignPublicIp = "ENABLED"
    }
  })
}
