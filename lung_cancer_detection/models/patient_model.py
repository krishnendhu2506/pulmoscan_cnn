from dataclasses import dataclass
from typing import Optional, Tuple, List

YES_NO = ("Yes", "No")
AIR_POLLUTION_LEVELS = ("Low", "Medium", "High")
OCCUPATIONAL_EXPOSURE = ("Mining", "Chemicals", "Dust", "None")


@dataclass
class PatientInput:
    name: str
    age: int
    gender: str
    contact: str
    notes: str
    smoking_history: str
    years_of_smoking: Optional[int]
    family_history_lung_cancer: str
    air_pollution_exposure: str
    occupational_exposure: str
    persistent_cough: str
    unexplained_weight_loss: str


def _clean_text(value: Optional[str]) -> str:
    return (value or "").strip()


def _normalize_choice(value: Optional[str], allowed: tuple[str, ...]) -> Optional[str]:
    cleaned = _clean_text(value)
    return cleaned if cleaned in allowed else None


def _parse_int(value: Optional[str]) -> Optional[int]:
    cleaned = _clean_text(value)
    if cleaned == "":
        return None
    if not cleaned.isdigit():
        return None
    return int(cleaned)


def parse_patient_form(form) -> Tuple[Optional[PatientInput], List[str]]:
    errors: List[str] = []

    name = _clean_text(form.get("name"))
    age = _parse_int(form.get("age"))
    gender = _clean_text(form.get("gender"))
    contact = _clean_text(form.get("contact"))
    notes = _clean_text(form.get("notes"))

    smoking_history = _normalize_choice(form.get("smoking_history"), YES_NO)
    years_of_smoking = _parse_int(form.get("years_of_smoking"))
    family_history = _normalize_choice(form.get("family_history_lung_cancer"), YES_NO)
    air_pollution = _normalize_choice(form.get("air_pollution_exposure"), AIR_POLLUTION_LEVELS)
    occupational = _normalize_choice(form.get("occupational_exposure"), OCCUPATIONAL_EXPOSURE)
    persistent_cough = _normalize_choice(form.get("persistent_cough"), YES_NO)
    unexplained_weight_loss = _normalize_choice(form.get("unexplained_weight_loss"), YES_NO)

    if not name:
        errors.append("Name is required.")
    if age is None or age < 0:
        errors.append("Age must be a non-negative number.")
    if not gender:
        errors.append("Gender is required.")

    if smoking_history is None:
        errors.append("Smoking history must be Yes or No.")
    if family_history is None:
        errors.append("Family history must be Yes or No.")
    if air_pollution is None:
        errors.append("Air pollution exposure must be Low, Medium, or High.")
    if occupational is None:
        errors.append("Occupational exposure must be one of the listed options.")
    if persistent_cough is None:
        errors.append("Persistent cough must be Yes or No.")
    if unexplained_weight_loss is None:
        errors.append("Unexplained weight loss must be Yes or No.")

    if smoking_history == "Yes":
        if years_of_smoking is None:
            errors.append("Years of smoking is required when smoking history is Yes.")
        elif years_of_smoking < 0:
            errors.append("Years of smoking must be a non-negative number.")

    if errors:
        return None, errors

    return (
        PatientInput(
            name=name,
            age=age,
            gender=gender,
            contact=contact,
            notes=notes,
            smoking_history=smoking_history or "No",
            years_of_smoking=years_of_smoking,
            family_history_lung_cancer=family_history or "No",
            air_pollution_exposure=air_pollution or "Low",
            occupational_exposure=occupational or "None",
            persistent_cough=persistent_cough or "No",
            unexplained_weight_loss=unexplained_weight_loss or "No",
        ),
        [],
    )
