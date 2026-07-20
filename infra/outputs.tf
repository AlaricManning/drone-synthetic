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
