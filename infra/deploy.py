import argparse
from google.cloud import aiplatform


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--image", required=True)
    args = p.parse_args()

    aiplatform.init(project=args.project, location=args.region)
    artifact_uri = f"gs://{args.bucket}/models/v1"

    model = aiplatform.Model.upload(
        display_name="telco-churn",
        artifact_uri=artifact_uri,
        serving_container_image_uri=args.image,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        version_aliases=["candidate"],
    )
    model.wait()

    endpoint = aiplatform.Endpoint.create(
        display_name="telco-churn-endpoint"
    )
    endpoint.deploy(
        model=model,
        machine_type="e2-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
        sync=True,
    )
    print("MODEL:", model.resource_name)
    print("ENDPOINT:", endpoint.resource_name)


if __name__ == "__main__":
    main()
