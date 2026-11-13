"""
backend/tests/test_db.py
========================
Phase 6.6 database service-layer tests.

Tests are purely service-layer: they create a real SQLAlchemy session bound to
TEST_DATABASE_URL (diabetes_risk_test_db) and exercise CRUD operations without
going through HTTP (avoids Python 3.14 anyio loopback socket restriction in sandbox).

All tests are idempotent: they run against the test DB, which is isolated from
the production database.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.core.database import Base
from backend.app.schemas.db_schemas import AssessmentCreate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_DB_URL = settings.TEST_DATABASE_URL or settings.DATABASE_URL.replace(
    "diabetes_risk_db", "diabetes_risk_test_db"
)


@pytest.fixture(scope="module")
def test_engine():
    """Create a fresh engine bound to the test database."""
    engine = create_engine(TEST_DB_URL, pool_pre_ping=True, echo=False)
    # Import models to register metadata
    from backend.app.models import user, assessment, prediction  # noqa: F401

    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="module")
def TestSessionLocal(test_engine):
    return sessionmaker(bind=test_engine, autocommit=False, autoflush=False)


@pytest.fixture()
def db(TestSessionLocal):
    """Function-scoped transactional DB session — rolls back after each test."""
    session = TestSessionLocal()
    try:
        yield session
        session.rollback()
    finally:
        session.close()


def _sample_assessment_data(**overrides) -> AssessmentCreate:
    """Return a valid 21-feature CDC assessment payload."""
    defaults = dict(
        HighBP=1, HighChol=0, CholCheck=1, BMI=27.5,
        Smoker=0, Stroke=0, HeartDiseaseorAttack=0,
        PhysActivity=1, Fruits=1, Veggies=1,
        HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0,
        GenHlth=3, MentHlth=0, PhysHlth=0,
        DiffWalk=0, Sex=1, Age=7, Education=5, Income=6,
        user_id=None,
    )
    defaults.update(overrides)
    return AssessmentCreate(**defaults)


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

class TestDatabaseConnectivity:
    def test_test_database_reachable(self, test_engine):
        """Test DB must be reachable via SQLAlchemy engine."""
        with test_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_tables_created(self, test_engine):
        """All three application tables must exist in the test DB."""
        with test_engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name IN "
                    "('users', 'assessments', 'predictions')"
                )
            )
            tables = {row[0] for row in result}
        assert tables == {"users", "assessments", "predictions"}


# ---------------------------------------------------------------------------
# Assessment CRUD
# ---------------------------------------------------------------------------

class TestAssessmentCRUD:
    def test_create_assessment_returns_id(self, db):
        from backend.app.services import db_service

        data = _sample_assessment_data()
        assessment = db_service.create_assessment(db, data)
        db.flush()

        assert assessment.id is not None
        assert assessment.id > 0

    def test_create_assessment_persists_features(self, db):
        from backend.app.services import db_service

        data = _sample_assessment_data(BMI=35.0, Age=10, GenHlth=5)
        assessment = db_service.create_assessment(db, data)
        db.flush()

        fetched = db_service.get_assessment(db, assessment.id)
        assert fetched is not None
        assert fetched.BMI == pytest.approx(35.0, abs=0.01)
        assert fetched.Age == 10
        assert fetched.GenHlth == 5

    def test_create_assessment_guest_user_id_is_null(self, db):
        from backend.app.services import db_service

        data = _sample_assessment_data(user_id=None)
        assessment = db_service.create_assessment(db, data)
        db.flush()

        assert assessment.user_id is None

    def test_get_nonexistent_assessment_returns_none(self, db):
        from backend.app.services import db_service

        result = db_service.get_assessment(db, assessment_id=999_999_999)
        assert result is None

    def test_list_assessments_pagination(self, db):
        from backend.app.services import db_service

        # Create 5 assessments
        for _ in range(5):
            db_service.create_assessment(db, _sample_assessment_data())
        db.flush()

        total, items = db_service.list_assessments(db, page=1, page_size=3)
        assert total >= 5
        assert len(items) <= 3

    def test_list_assessments_page2(self, db):
        from backend.app.services import db_service

        for _ in range(6):
            db_service.create_assessment(db, _sample_assessment_data())
        db.flush()

        total, page1 = db_service.list_assessments(db, page=1, page_size=4)
        _, page2 = db_service.list_assessments(db, page=2, page_size=4)

        ids_p1 = {a.id for a in page1}
        ids_p2 = {a.id for a in page2}
        assert ids_p1.isdisjoint(ids_p2)


# ---------------------------------------------------------------------------
# Prediction CRUD
# ---------------------------------------------------------------------------

class TestPredictionCRUD:
    def _create_assessment(self, db):
        from backend.app.services import db_service
        a = db_service.create_assessment(db, _sample_assessment_data())
        db.flush()
        return a

    def test_create_prediction_returns_id(self, db):
        from backend.app.services import db_service

        a = self._create_assessment(db)
        pred = db_service.create_prediction(
            db=db,
            assessment_id=a.id,
            predicted_class=0,
            predicted_label="No Diabetes",
            risk_tier="Low / Baseline Risk",
            confidence=0.75,
            prob_no_diabetes=0.75,
            prob_prediabetes=0.15,
            prob_diabetes=0.10,
            model_version="LogisticRegression-GroupAware",
        )
        db.flush()

        assert pred.id is not None
        assert pred.id > 0

    def test_get_prediction_by_id(self, db):
        from backend.app.services import db_service

        a = self._create_assessment(db)
        pred = db_service.create_prediction(
            db=db, assessment_id=a.id, predicted_class=2,
            predicted_label="Diabetes", risk_tier="High Risk",
            confidence=0.85, prob_no_diabetes=0.05,
            prob_prediabetes=0.10, prob_diabetes=0.85,
            model_version="LogisticRegression-GroupAware",
        )
        db.flush()

        fetched = db_service.get_prediction(db, pred.id)
        assert fetched is not None
        assert fetched.predicted_class == 2
        assert fetched.probability_diabetes == pytest.approx(0.85, abs=1e-6)

    def test_get_prediction_by_assessment(self, db):
        from backend.app.services import db_service

        a = self._create_assessment(db)
        db_service.create_prediction(
            db=db, assessment_id=a.id, predicted_class=1,
            predicted_label="Prediabetes", risk_tier="Elevated Risk",
            confidence=0.60, prob_no_diabetes=0.25,
            prob_prediabetes=0.60, prob_diabetes=0.15,
            model_version="LogisticRegression-GroupAware",
        )
        db.flush()

        fetched = db_service.get_prediction_by_assessment(db, a.id)
        assert fetched is not None
        assert fetched.assessment_id == a.id

    def test_get_nonexistent_prediction_returns_none(self, db):
        from backend.app.services import db_service

        result = db_service.get_prediction(db, prediction_id=999_999_999)
        assert result is None

    def test_list_predictions_pagination(self, db):
        from backend.app.services import db_service

        for _ in range(5):
            a = self._create_assessment(db)
            db_service.create_prediction(
                db=db, assessment_id=a.id, predicted_class=0,
                predicted_label="No Diabetes", risk_tier="Low / Baseline Risk",
                confidence=0.70, prob_no_diabetes=0.70,
                prob_prediabetes=0.20, prob_diabetes=0.10,
                model_version="LogisticRegression-GroupAware",
            )
        db.flush()

        total, items = db_service.list_predictions(db, page=1, page_size=3)
        assert total >= 5
        assert len(items) <= 3


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------

class TestDashboardStats:
    def test_stats_returns_expected_keys(self, db):
        from backend.app.services import db_service

        stats = db_service.get_dashboard_stats(db)
        assert "total_assessments" in stats
        assert "total_predictions" in stats
        assert "class_distribution" in stats
        assert "recent_assessments" in stats

    def test_class_distribution_has_all_classes(self, db):
        from backend.app.services import db_service

        stats = db_service.get_dashboard_stats(db)
        dist = stats["class_distribution"]
        assert 0 in dist
        assert 1 in dist
        assert 2 in dist

    def test_stats_counts_increase_after_insert(self, db):
        from backend.app.services import db_service

        before = db_service.get_dashboard_stats(db)
        before_count = before["total_assessments"]

        db_service.create_assessment(db, _sample_assessment_data())
        db.flush()

        after = db_service.get_dashboard_stats(db)
        assert after["total_assessments"] == before_count + 1

    def test_recent_assessments_max_five(self, db):
        from backend.app.services import db_service

        stats = db_service.get_dashboard_stats(db)
        assert len(stats["recent_assessments"]) <= 5


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_bmi_out_of_range_raises_validation_error(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AssessmentCreate(
                HighBP=1, HighChol=0, CholCheck=1, BMI=200.0,  # invalid
                Smoker=0, Stroke=0, HeartDiseaseorAttack=0,
                PhysActivity=1, Fruits=1, Veggies=1,
                HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0,
                GenHlth=3, MentHlth=0, PhysHlth=0,
                DiffWalk=0, Sex=1, Age=7, Education=5, Income=6,
            )

    def test_age_out_of_range_raises_validation_error(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AssessmentCreate(
                HighBP=1, HighChol=0, CholCheck=1, BMI=27.5,
                Smoker=0, Stroke=0, HeartDiseaseorAttack=0,
                PhysActivity=1, Fruits=1, Veggies=1,
                HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0,
                GenHlth=3, MentHlth=0, PhysHlth=0,
                DiffWalk=0, Sex=1, Age=99,  # invalid (1–13)
                Education=5, Income=6,
            )

    def test_binary_feature_out_of_range_raises_validation_error(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AssessmentCreate(
                HighBP=5,  # invalid (0 or 1)
                HighChol=0, CholCheck=1, BMI=27.5,
                Smoker=0, Stroke=0, HeartDiseaseorAttack=0,
                PhysActivity=1, Fruits=1, Veggies=1,
                HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0,
                GenHlth=3, MentHlth=0, PhysHlth=0,
                DiffWalk=0, Sex=1, Age=7, Education=5, Income=6,
            )

