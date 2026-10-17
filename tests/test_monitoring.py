from src.final_project.app import metrics, record_prediction_metrics


def test_metrics_endpoint_exposes_prometheus_format():
    response = metrics()

    assert response.status_code == 200
    body = response.body.decode()
    assert "pets_api_http_requests_total" in body
    assert "pets_api_predictions_total" in body
    assert "pets_api_prediction_confidence_bucket" in body


def test_prediction_metrics_record_confidence_and_decision():
    record_prediction_metrics(0.87, "confident")
    body = metrics().body.decode()

    assert 'pets_api_predictions_total{decision="confident"} 1' in body
    assert "pets_api_prediction_confidence_count 1" in body
