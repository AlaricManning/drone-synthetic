# The build identity: what assembles many converted runs into one dataset
# version. This is the "build role" row of the README's security model table.
#
# A distinct role rather than reusing the convert job role, for two reasons.
# Conversion has no reason to read datasets/*, so sharing a role would
# over-grant it. And a build *does* need that read, to pull the annotations and
# provenance its inputs were converted into — so a build running as the convert
# role would fail on AccessDenied while the identical code succeeded locally
# under admin. Which is the failure mode worth designing out: if local builds
# use one identity and Batch builds another, "it worked locally" stops
# predicting "it works on Batch".
#
# Two trusted principals, deliberately:
#
#   ecs-tasks   so a Batch build job runs with exactly these grants
#   named human so a one-off local build is an assume-role rather than a fifth
#               long-lived access key — temporary credentials, the same grants
#               as Batch, and CloudTrail attributing each build to a session
#
# Notably absent: s3:ListBucket. A build is told which runs it is assembling, so
# it never enumerates the bucket; every key it reads is one it derived from its
# config. Nothing here can discover data it was not pointed at.

locals {
  # Falling back to whoever is applying, rather than to nobody. An apply that
  # forgot the variable would otherwise silently strip local build access, and
  # the operator applying terraform is the same person who runs one-off builds
  # — this is a single-operator setup, as versions.tf notes.
  build_principals = length(var.build_role_principals) > 0 ? var.build_role_principals : [
    data.aws_caller_identity.current.arn
  ]
}

data "aws_iam_policy_document" "build_assume" {
  statement {
    sid     = "BatchBuildJobs"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }

  statement {
    sid     = "NamedOperators"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "AWS"
      identifiers = local.build_principals
    }
  }
}

resource "aws_iam_role" "build" {
  name               = "dronesynth-build"
  description        = "Assembles converted runs into a versioned dataset"
  assume_role_policy = data.aws_iam_policy_document.build_assume.json
}

resource "aws_iam_role_policy" "build" {
  name = "build-datasets-from-runs"
  role = aws_iam_role.build.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # raw/* for the frames and run manifests; datasets/* for the annotations
        # and converter provenance the build reads back out of each per-run
        # conversion. Server-side CopyObject needs GetObject on the source, which
        # is what keeps the frames from travelling through the builder.
        Sid    = "ReadRunsAndConversions"
        Effect = "Allow"
        Action = ["s3:GetObject"]
        Resource = [
          "${aws_s3_bucket.data.arn}/raw/*",
          "${aws_s3_bucket.data.arn}/datasets/*",
        ]
      },
      {
        Sid      = "WriteDatasetVersions"
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.data.arn}/datasets/*"
      },
    ]
  })
}
