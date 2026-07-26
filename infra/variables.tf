variable "aws_region" {
  description = "Region for all pipeline resources"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "S3 bucket that is the system of record (raw runs, datasets, QC)"
  type        = string
}

variable "build_role_principals" {
  description = <<-EOT
    IAM user/role ARNs allowed to assume the dataset build role, for one-off
    builds run outside Batch. Defaults to whoever applies terraform, so local
    builds keep working without having to name yourself on every apply.
  EOT
  type        = list(string)
  default     = []
}
