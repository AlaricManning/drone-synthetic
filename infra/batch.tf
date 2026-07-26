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

# One image, two job definitions. The image's entrypoint is the bare CLI, so the
# subcommand and its config live here, per job, rather than being baked in.
#
# Submitters fill in Ref:: placeholders through submit_job's `parameters` instead
# of overriding `command`. That is deliberate: a command override *replaces* the
# command outright, so anything the job definition put there — the subcommand,
# the config path — would vanish the moment a submitter passed a run id.
# Parameters keep a submission to what only it knows, which is what makes the
# job definition the single place the image and its arguments are decided.

resource "aws_batch_job_definition" "convert" {
  name                  = "dronesynth-convert"
  type                  = "container"
  platform_capabilities = ["FARGATE"]

  parameters = {
    run_id  = "unset"
    version = "unset"
  }

  container_properties = jsonencode({
    image = "${aws_ecr_repository.convert.repository_url}:latest"
    command = [
      "convert",
      "--config", "configs/convert.s3.yaml",
      "--run-id", "Ref::run_id",
      "--version", "Ref::version",
    ]
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

# Builds assemble a dataset version from already-converted runs. Separate from
# conversion because it is a different job with different credentials: the build
# role reads raw and datasets and writes only datasets, and never lists either.
#
# The config is a parameter rather than fixed because there is one build config
# per dataset version, all of them baked into the image; which version this job
# is assembling is precisely the thing the submitter knows.
resource "aws_batch_job_definition" "build" {
  name                  = "dronesynth-build"
  type                  = "container"
  platform_capabilities = ["FARGATE"]

  parameters = {
    config  = "unset"
    version = "unset"
  }

  container_properties = jsonencode({
    image = "${aws_ecr_repository.convert.repository_url}:latest"
    command = [
      "build",
      "--config", "Ref::config",
      "--version", "Ref::version",
    ]
    resourceRequirements = [
      { type = "VCPU", value = "1" },
      { type = "MEMORY", value = "2048" },
    ]
    executionRoleArn = aws_iam_role.batch_execution.arn
    jobRoleArn       = aws_iam_role.build.arn
    networkConfiguration = {
      assignPublicIp = "ENABLED"
    }
  })
}
