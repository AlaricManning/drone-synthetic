# The ingest identity: the only credential the capture machine holds.
# Put-only on raw/* — it cannot list the bucket, read objects back, or delete
# anything, so a leaked key cannot enumerate or destroy captured data.

resource "aws_iam_user" "ingest" {
  name = "drone-synth-ingest"
}

resource "aws_iam_user_policy" "ingest_put_only" {
  name = "put-only-raw"
  user = aws_iam_user.ingest.name

  policy = jsonencode({
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

# NOTE: no aws_iam_access_key resource on purpose — terraform state would
# store the secret in plaintext. Create the key manually after apply:
#   aws iam create-access-key --user-name drone-synth-ingest
# and put it in ~/.aws/credentials under a profile, never in this repo.
