terraform {
  required_version = ">= 1.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # local state while the footprint is one bucket and a handful of roles;
  # move to an S3 backend if this ever has more than one operator
}

provider "aws" {
  region = var.aws_region
}
