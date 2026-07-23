# Manifest lands in raw/ -> EventBridge -> Lambda -> Batch job.
# The Lambda exists because EventBridge cannot parse the run id out of the
# S3 key on its own; it does exactly that and calls SubmitJob.

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket_notification" "eventbridge" {
  bucket      = aws_s3_bucket.data.id
  eventbridge = true
}

data "archive_file" "auto_trigger" {
  type        = "zip"
  source_file = "${path.module}/lambda/auto_trigger.py"
  output_path = "${path.module}/.build/auto_trigger.zip"
}

resource "aws_iam_role" "auto_trigger" {
  name = "dronesynth-auto-trigger"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Action    = "sts:AssumeRole"
        Principal = { Service = "lambda.amazonaws.com" }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "auto_trigger_logs" {
  role       = aws_iam_role.auto_trigger.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "auto_trigger_submit" {
  name = "submit-conversion-jobs"
  role = aws_iam_role.auto_trigger.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SubmitConversionJobs"
        Effect = "Allow"
        Action = ["batch:SubmitJob"]
        Resource = [
          aws_batch_job_queue.convert.arn,
          # both ARN forms: submitting by bare name is checked against the
          # revisionless ARN, submitting a pinned revision against :N
          "arn:aws:batch:${var.aws_region}:${data.aws_caller_identity.current.account_id}:job-definition/${aws_batch_job_definition.convert.name}",
          "arn:aws:batch:${var.aws_region}:${data.aws_caller_identity.current.account_id}:job-definition/${aws_batch_job_definition.convert.name}:*",
        ]
      }
    ]
  })
}

resource "aws_lambda_function" "auto_trigger" {
  function_name    = "dronesynth-auto-trigger"
  role             = aws_iam_role.auto_trigger.arn
  runtime          = "python3.12"
  handler          = "auto_trigger.handler"
  filename         = data.archive_file.auto_trigger.output_path
  source_code_hash = data.archive_file.auto_trigger.output_base64sha256
  timeout          = 30

  environment {
    variables = {
      JOB_QUEUE      = aws_batch_job_queue.convert.name
      JOB_DEFINITION = aws_batch_job_definition.convert.name
    }
  }
}

resource "aws_cloudwatch_event_rule" "manifest_created" {
  name        = "dronesynth-manifest-created"
  description = "A run's manifest landed in raw/ — the run is complete, convert it"

  event_pattern = jsonencode({
    source      = ["aws.s3"]
    detail-type = ["Object Created"]
    detail = {
      bucket = { name = [aws_s3_bucket.data.bucket] }
      object = { key = [{ wildcard = "raw/*/manifest.json" }] }
    }
  })
}

resource "aws_cloudwatch_event_target" "auto_trigger" {
  rule = aws_cloudwatch_event_rule.manifest_created.name
  arn  = aws_lambda_function.auto_trigger.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.auto_trigger.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.manifest_created.arn
}

output "auto_trigger_function" {
  description = "Lambda that submits conversion jobs when manifests land"
  value       = aws_lambda_function.auto_trigger.function_name
}
