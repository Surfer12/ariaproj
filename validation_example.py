#!/usr/bin/env python3
"""
Validation Example for Improved Hybrid Reasoning System
Demonstrates how the enhanced YAML tag definitions enable robust validation
"""

import yaml
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class ValidationResult:
    """Results of validation checks"""
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: bool = True
    
    def add_error(self, message: str):
        self.errors.append(message)
        self.passed = False
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def report(self):
        if self.passed:
            print("✅ Validation PASSED")
        else:
            print("❌ Validation FAILED")
        
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

class RecursionSafetyValidator:
    """Validates recursion safety for the MetaOptimizationController"""
    
    @staticmethod
    def validate_recursion_depth_limit(config: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()
        
        if "recursion_depth_limit" not in config:
            result.add_error("Missing recursion_depth_limit")
            return result
        
        limit_config = config["recursion_depth_limit"]
        
        # Check for required safety parameters
        required_fields = ["max_iterations", "convergence_threshold", "timeout_minutes", "termination_conditions"]
        for field in required_fields:
            if field not in limit_config:
                result.add_error(f"Missing required field: {field}")
        
        # Validate max_iterations
        if "max_iterations" in limit_config:
            max_iter = limit_config["max_iterations"]
            if not isinstance(max_iter, int) or max_iter <= 0:
                result.add_error("max_iterations must be a positive integer")
            elif max_iter > 1000:
                result.add_warning(f"max_iterations ({max_iter}) is very high, may cause long execution times")
        
        # Validate convergence_threshold
        if "convergence_threshold" in limit_config:
            threshold = limit_config["convergence_threshold"]
            if not isinstance(threshold, (int, float)) or threshold <= 0:
                result.add_error("convergence_threshold must be a positive number")
        
        # Validate timeout
        if "timeout_minutes" in limit_config:
            timeout = limit_config["timeout_minutes"]
            if not isinstance(timeout, int) or timeout <= 0:
                result.add_error("timeout_minutes must be a positive integer")
        
        # Validate termination conditions
        if "termination_conditions" in limit_config:
            conditions = limit_config["termination_conditions"]
            required_conditions = ["convergence_achieved", "max_iterations_reached", "timeout_exceeded", "user_intervention"]
            for condition in required_conditions:
                if condition not in conditions:
                    result.add_warning(f"Missing recommended termination condition: {condition}")
        
        return result

class SchemaValidator:
    """Validates input/output schemas for modules"""
    
    @staticmethod
    def validate_schema(schema: Dict[str, Any], data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()
        
        if schema.get("type") != "object":
            result.add_error("Schema must be of type 'object'")
            return result
        
        properties = schema.get("properties", {})
        
        # Check required fields
        for prop_name, prop_def in properties.items():
            if prop_def.get("required", False) and prop_name not in data:
                result.add_error(f"Missing required field: {prop_name}")
        
        # Validate present fields
        for field_name, field_value in data.items():
            if field_name not in properties:
                result.add_warning(f"Unexpected field: {field_name}")
                continue
            
            prop_def = properties[field_name]
            expected_type = prop_def.get("type")
            
            # Type validation
            if expected_type == "string" and not isinstance(field_value, str):
                result.add_error(f"Field '{field_name}' must be a string")
            elif expected_type == "integer" and not isinstance(field_value, int):
                result.add_error(f"Field '{field_name}' must be an integer")
            elif expected_type == "float" and not isinstance(field_value, (int, float)):
                result.add_error(f"Field '{field_name}' must be a number")
            elif expected_type == "boolean" and not isinstance(field_value, bool):
                result.add_error(f"Field '{field_name}' must be a boolean")
            elif expected_type == "array" and not isinstance(field_value, list):
                result.add_error(f"Field '{field_name}' must be an array")
            
            # Range validation
            if "range" in prop_def:
                range_min, range_max = prop_def["range"]
                if isinstance(field_value, (int, float)):
                    if field_value < range_min or field_value > range_max:
                        result.add_error(f"Field '{field_name}' value {field_value} is outside range [{range_min}, {range_max}]")
            
            # Enum validation
            if "enum" in prop_def:
                if field_value not in prop_def["enum"]:
                    result.add_error(f"Field '{field_name}' value '{field_value}' not in allowed values: {prop_def['enum']}")
        
        return result

class DefaultValueValidator:
    """Validates that default values are reasonable and consistent"""
    
    @staticmethod
    def validate_defaults(tag_config: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()
        
        attributes = tag_config.get("attributes", [])
        
        for attr in attributes:
            if "default_value" not in attr:
                result.add_warning(f"Attribute '{attr.get('name')}' missing default value")
                continue
            
            default_val = attr["default_value"]
            attr_type = attr.get("type")
            
            # Type consistency
            if attr_type == "string" and not isinstance(default_val, str):
                result.add_error(f"Default value for '{attr.get('name')}' should be string")
            elif attr_type == "integer" and not isinstance(default_val, int):
                result.add_error(f"Default value for '{attr.get('name')}' should be integer")
            elif attr_type == "float" and not isinstance(default_val, (int, float)):
                result.add_error(f"Default value for '{attr.get('name')}' should be number")
            elif attr_type == "boolean" and not isinstance(default_val, bool):
                result.add_error(f"Default value for '{attr.get('name')}' should be boolean")
            elif attr_type == "array" and not isinstance(default_val, list):
                result.add_error(f"Default value for '{attr.get('name')}' should be array")
            
            # Range validation
            if "range" in attr:
                range_min, range_max = attr["range"]
                if isinstance(default_val, (int, float)):
                    if default_val < range_min or default_val > range_max:
                        result.add_error(f"Default value for '{attr.get('name')}' is outside valid range")
            
            # Enum validation
            if "enum" in attr:
                if default_val not in attr["enum"]:
                    result.add_error(f"Default value for '{attr.get('name')}' not in allowed enum values")
        
        return result

class ErrorHandlingValidator:
    """Validates error handling configurations"""
    
    @staticmethod
    def validate_error_config(tag_config: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()
        
        # Check for failure modes
        if "failure_modes" not in tag_config:
            result.add_warning("No failure modes specified")
        else:
            failure_modes = tag_config["failure_modes"]
            if not isinstance(failure_modes, list) or not failure_modes:
                result.add_error("failure_modes must be a non-empty list")
        
        # Check for error handling config
        if "error_handling" not in tag_config:
            result.add_warning("No error handling configuration")
        else:
            error_config = tag_config["error_handling"]
            
            # Validate timeout
            if "timeout_seconds" in error_config:
                timeout = error_config["timeout_seconds"]
                if not isinstance(timeout, int) or timeout <= 0:
                    result.add_error("timeout_seconds must be a positive integer")
            
            # Validate retries
            if "max_retries" in error_config:
                retries = error_config["max_retries"]
                if not isinstance(retries, int) or retries < 0:
                    result.add_error("max_retries must be a non-negative integer")
            
            # Validate fallback strategy
            if "fallback_strategy" in error_config:
                strategy = error_config["fallback_strategy"]
                valid_strategies = ["graceful_degradation", "error_propagation", "safe_abort"]
                if strategy not in valid_strategies:
                    result.add_error(f"Invalid fallback_strategy: {strategy}")
        
        return result

def validate_system_config(config_file: str) -> ValidationResult:
    """Main validation function for the entire system"""
    overall_result = ValidationResult()
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        overall_result.add_error(f"Failed to load config file: {e}")
        return overall_result
    
    # Validate system metadata
    if "system_metadata" not in config:
        overall_result.add_warning("Missing system metadata")
    
    # Validate each tag
    tags = config.get("tags", [])
    for tag in tags:
        tag_name = tag.get("name", "unknown")
        print(f"Validating tag: {tag_name}")
        
        # Validate recursion safety for MetaOptimizationController
        if tag_name == "MetaOptimizationController":
            recursion_result = RecursionSafetyValidator.validate_recursion_depth_limit(tag)
            overall_result.errors.extend(recursion_result.errors)
            overall_result.warnings.extend(recursion_result.warnings)
            if not recursion_result.passed:
                overall_result.passed = False
        
        # Validate default values
        defaults_result = DefaultValueValidator.validate_defaults(tag)
        overall_result.errors.extend(defaults_result.errors)
        overall_result.warnings.extend(defaults_result.warnings)
        if not defaults_result.passed:
            overall_result.passed = False
        
        # Validate error handling
        error_result = ErrorHandlingValidator.validate_error_config(tag)
        overall_result.errors.extend(error_result.errors)
        overall_result.warnings.extend(error_result.warnings)
        if not error_result.passed:
            overall_result.passed = False
    
    return overall_result

def demonstrate_schema_validation():
    """Demonstrate schema validation with example data"""
    print("=== Schema Validation Demo ===")
    
    # Example SymbolicModule input schema
    symbolic_input_schema = {
        "type": "object",
        "properties": {
            "problem_statement": {"type": "string", "required": True},
            "constraints": {"type": "array"},
            "timeout_ms": {"type": "integer", "default": 30000, "range": [1000, 300000]}
        }
    }
    
    # Valid input data
    valid_data = {
        "problem_statement": "Prove that P implies Q",
        "constraints": ["P", "not Q implies contradiction"],
        "timeout_ms": 60000
    }
    
    # Invalid input data
    invalid_data = {
        "problem_statement": 123,  # Should be string
        "timeout_ms": 500000  # Outside range
        # Missing required field
    }
    
    print("\n--- Valid Data ---")
    valid_result = SchemaValidator.validate_schema(symbolic_input_schema, valid_data)
    valid_result.report()
    
    print("\n--- Invalid Data ---")
    invalid_result = SchemaValidator.validate_schema(symbolic_input_schema, invalid_data)
    invalid_result.report()

def main():
    """Main demonstration function"""
    print("🔍 Hybrid Reasoning System - Validation Demo")
    print("=" * 50)
    
    # Demonstrate schema validation
    demonstrate_schema_validation()
    
    # Validate the main config file if it exists
    config_file = "improved_hybrid_reasoning_system.yaml"
    try:
        print(f"\n=== Validating Configuration File: {config_file} ===")
        result = validate_system_config(config_file)
        result.report()
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found - skipping full validation")
    
    print("\n🎯 Validation Benefits Demonstrated:")
    print("  ✅ Recursion safety enforcement")
    print("  ✅ Type safety validation")
    print("  ✅ Range and enum checking")
    print("  ✅ Required field validation")
    print("  ✅ Error handling verification")
    print("  ✅ Default value consistency")

if __name__ == "__main__":
    main()