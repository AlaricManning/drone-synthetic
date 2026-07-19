variable "aws_region" {
  description = "Region for all pipeline resources"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "S3 bucket that is the system of record (raw runs, datasets, QC)"
  type        = string
}
