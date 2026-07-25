# The capture identities: the only credentials the machines producing runs hold.
# Put-only on raw/* — they cannot list the bucket, read objects back, or delete
# anything, so a leaked key cannot enumerate or destroy captured data.
#
# Two users rather than one shared key: either can be rotated or revoked without
# interrupting the other, and CloudTrail attributes every write under raw/ to the
# machine that made it.
#
#   drone-synth-ingest  operator running `dronesynth ingest` on EasySynth output
#   drone-synth-render  the Unreal render box publishing runs by itself, from the
#                       drone-synth-render repo

locals {
  # Both identities get the identical grant; keeping it in one place is what
  # makes "put-only on raw/*" a property of the system rather than of a user.
  put_only_raw_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "PutRawRunsOnly"
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.data.arn}/raw/*"
      }
    ]
  })
}

resource "aws_iam_user" "ingest" {
  name = "drone-synth-ingest"
}

resource "aws_iam_user_policy" "ingest_put_only" {
  name   = "put-only-raw"
  user   = aws_iam_user.ingest.name
  policy = local.put_only_raw_policy
}

resource "aws_iam_user" "render" {
  name = "drone-synth-render"
}

resource "aws_iam_user_policy" "render_put_only" {
  name   = "put-only-raw"
  user   = aws_iam_user.render.name
  policy = local.put_only_raw_policy
}

# NOTE: no aws_iam_access_key resources on purpose — terraform state would
# store the secrets in plaintext. Create a key manually after apply:
#   aws iam create-access-key --user-name drone-synth-ingest
#   aws iam create-access-key --user-name drone-synth-render
# and put it in ~/.aws/credentials under a profile, never in this repo.
