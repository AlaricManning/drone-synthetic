output "bucket_name" {
  description = "System-of-record bucket"
  value       = aws_s3_bucket.data.bucket
}

output "raw_root" {
  description = "Storage root for configs: where ingest writes runs"
  value       = "s3://${aws_s3_bucket.data.bucket}/raw"
}

output "ingest_user" {
  description = "Put-only IAM user for the capture machine"
  value       = aws_iam_user.ingest.name
}

output "ecr_repository_url" {
  description = "Where the conversion image gets pushed"
  value       = aws_ecr_repository.convert.repository_url
}

output "job_queue" {
  description = "Batch queue dronesynth submit targets"
  value       = aws_batch_job_queue.convert.name
}

output "job_definition" {
  description = "Batch job definition for conversion"
  value       = aws_batch_job_definition.convert.name
}
