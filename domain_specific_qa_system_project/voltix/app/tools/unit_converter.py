from typing import Dict, Any
from app.utils.logger import get_logger

logger = get_logger("voltix.unit_converter")

# Mapping of common shorthand EEE unit symbols to standard Pint representations
EEE_UNIT_ALIASES = {
    "kw": "kilowatt",
    "w": "watt",
    "mw": "megawatt",
    "a": "ampere",
    "ma": "milliampere",
    "ua": "microampere",
    "v": "volt",
    "kv": "kilovolt",
    "mv": "millivolt",
    "rpm": "rpm",
    "rad/s": "rad/s",
    "rad_s": "rad/s",
    "wh": "watt_hour",
    "kwh": "kilowatt_hour",
    "mwh": "megawatt_hour",
    "ah": "ampere_hour",
    "mah": "milliampere_hour",
    "ohm": "ohm",
    "kohm": "kiloohm",
    "mohm": "megaohm",
    "uf": "microfarad",
    "nf": "nanofarad",
    "pf": "picofarad",
    "f": "farad",
    "mh": "millihenry",
    "uh": "microhenry",
    "h": "henry",
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",
    "hp": "horsepower",
    "nm": "newton_meter"
}

class UnitConverter:
    """Robust Engineering unit converter for Electrical and Electronics Engineering parameters."""

    @staticmethod
    def convert(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        try:
            import pint
            ureg = pint.UnitRegistry()

            # Normalize alias names
            from_clean = from_unit.strip().lower()
            to_clean = to_unit.strip().lower()

            from_standard = EEE_UNIT_ALIASES.get(from_clean, from_unit.strip())
            to_standard = EEE_UNIT_ALIASES.get(to_clean, to_unit.strip())

            val_quantity = value * ureg(from_standard)
            converted = val_quantity.to(to_standard)

            return {
                "success": True,
                "input_value": value,
                "input_unit": from_unit,
                "result_value": round(float(converted.magnitude), 6),
                "result_unit": to_unit,
                "formatted": f"{value} {from_unit} = {round(float(converted.magnitude), 6)} {to_unit}"
            }
        except Exception as e:
            logger.error(f"Unit conversion failed ({value} {from_unit} -> {to_unit}): {e}")
            return {
                "success": False,
                "error": f"Incompatible or unsupported unit conversion from '{from_unit}' to '{to_unit}'."
            }
