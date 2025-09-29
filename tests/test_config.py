# test_config.py
"""Test module for configuration constants and validation."""

import pytest
import re
from astrakairos import config


class TestPhysicalValidationRanges:
    """Test physical validation constants."""
    
    def test_period_ranges(self):
        """Test orbital period validation ranges."""
        assert config.MIN_PERIOD_YEARS > 0
        assert config.MAX_PERIOD_YEARS > config.MIN_PERIOD_YEARS
        assert config.MIN_PERIOD_YEARS == 0.1
        assert config.MAX_PERIOD_YEARS == 100000.0
    
    def test_semimajor_axis_ranges(self):
        """Test semi-major axis validation ranges."""
        assert config.MIN_SEMIMAJOR_AXIS_ARCSEC > 0
        assert config.MAX_SEMIMAJOR_AXIS_ARCSEC > config.MIN_SEMIMAJOR_AXIS_ARCSEC
        assert config.MIN_SEMIMAJOR_AXIS_ARCSEC == 0.001
        assert config.MAX_SEMIMAJOR_AXIS_ARCSEC == 100.0
    
    def test_eccentricity_ranges(self):
        """Test eccentricity validation ranges."""
        assert config.MIN_ECCENTRICITY == 0.0
        assert config.MAX_ECCENTRICITY < 1.0
        assert config.MAX_ECCENTRICITY == 0.99
        assert config.MAX_ECCENTRICITY > config.MIN_ECCENTRICITY
    
    def test_inclination_ranges(self):
        """Test inclination validation ranges."""
        assert config.MIN_INCLINATION_DEG == 0.0
        assert config.MAX_INCLINATION_DEG == 180.0
        assert config.MAX_INCLINATION_DEG > config.MIN_INCLINATION_DEG


class TestOrb6Configuration:
    """Test ORB6 catalog configuration."""
    
    def test_orb6_error_validation_thresholds(self):
        """Test ORB6 error validation thresholds are reasonable."""
        thresholds = config.ORB6_ERROR_VALIDATION_THRESHOLDS
        
        assert isinstance(thresholds, dict)
        expected_keys = ['e_P', 'e_a', 'e_i', 'e_Omega', 'e_T', 'e_e', 'e_omega_arg']
        
        for key in expected_keys:
            assert key in thresholds
            assert thresholds[key] > 0
        
        # Specific reasonable ranges
        assert thresholds['e_P'] >= 100.0  # Period errors
        assert thresholds['e_a'] >= 1.0    # Semi-major axis errors
        assert thresholds['e_i'] <= 180.0  # Inclination errors
        assert thresholds['e_e'] <= 1.0    # Eccentricity errors
    
    def test_orb6_fallback_errors(self):
        """Test ORB6 fallback error estimates."""
        fallbacks = config.ORB6_FALLBACK_ERRORS
        
        assert isinstance(fallbacks, dict)
        expected_keys = ['e_P', 'e_a', 'e_i', 'e_Omega', 'e_T', 'e_e', 'e_omega_arg']
        
        for key in expected_keys:
            assert key in fallbacks
            assert fallbacks[key] > 0
            # Fallback errors should be smaller than validation thresholds
            assert fallbacks[key] <= config.ORB6_ERROR_VALIDATION_THRESHOLDS[key]
    
    def test_wds_fallback_errors(self):
        """Test WDS fallback error estimates."""
        fallbacks = config.WDS_FALLBACK_ERRORS
        
        assert isinstance(fallbacks, dict)
        assert 'pa_error' in fallbacks
        assert 'sep_error' in fallbacks
        assert fallbacks['pa_error'] > 0
        assert fallbacks['sep_error'] > 0
    
    def test_orbit_grade_ranges(self):
        """Test orbital quality grade ranges."""
        assert config.MIN_ORBIT_GRADE == 1
        assert config.MAX_ORBIT_GRADE >= config.MIN_ORBIT_GRADE
        assert config.MAX_ORBIT_GRADE == 5


class TestCoordinateValidation:
    """Test coordinate validation constants."""
    
    def test_ra_ranges(self):
        """Test right ascension validation ranges."""
        assert config.MIN_RA_DEG == 0.0
        assert config.MAX_RA_DEG == 360.0
        assert config.MAX_RA_DEG > config.MIN_RA_DEG
    
    def test_dec_ranges(self):
        """Test declination validation ranges."""
        assert config.MIN_DEC_DEG == -90.0
        assert config.MAX_DEC_DEG == 90.0
        assert config.MAX_DEC_DEG > config.MIN_DEC_DEG
    
    def test_wds_designation_config(self):
        """Test WDS designation parsing configuration."""
        assert config.MIN_WDS_ID_LENGTH >= 10
        assert isinstance(config.WDS_COORDINATE_PATTERN, str)
        
        # Test that the regex pattern is valid
        pattern = re.compile(config.WDS_COORDINATE_PATTERN)
        
        # Test with valid WDS ID
        assert pattern.match("00013+1234")
        assert pattern.match("23591-8945AB")
        
        # Test with invalid WDS IDs
        assert not pattern.match("ABCDE+1234")
        assert not pattern.match("0001+123")


class TestMonteCarloConfiguration:
    """Test Monte Carlo simulation configuration."""
    
    def test_monte_carlo_parameters(self):
        """Test Monte Carlo simulation parameters."""
        assert config.DEFAULT_MC_SAMPLES > 0
        assert config.MC_CONFIDENCE_LEVEL > 0
        assert config.MC_CONFIDENCE_LEVEL < 100
        assert isinstance(config.MC_RANDOM_SEED, int)
        assert config.MC_RANDOM_SEED >= 0


class TestLoggingConfiguration:
    """Test logging configuration."""
    
    def test_logging_parameters(self):
        """Test logging configuration parameters."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert config.DEFAULT_LOG_LEVEL in valid_levels
        assert isinstance(config.DEFAULT_LOG_FORMAT, str)
        assert "%(asctime)s" in config.DEFAULT_LOG_FORMAT
    
    def test_validation_logging_config(self):
        """Test validation logging configuration."""
        assert isinstance(config.ENABLE_VALIDATION_WARNINGS, bool)
        assert 0.0 <= config.VALIDATION_WARNING_SAMPLE_RATE <= 1.0
        assert isinstance(config.ALLOW_SINGLE_EPOCH_SYSTEMS, bool)


class TestIOConfiguration:
    """Test I/O configuration parameters."""
    
    def test_csv_configuration(self):
        """Test CSV processing configuration."""
        assert config.CSV_SNIFFER_SAMPLE_SIZE > 0
        assert config.DEFAULT_COORDINATE_PRECISION >= 0
    
    def test_coordinate_error_behavior(self):
        """Test coordinate error handling configuration."""
        valid_behaviors = ["raise", "return_none", "return_invalid"]
        assert config.COORDINATE_ERROR_BEHAVIOR in valid_behaviors


class TestTechniqueErrorModel:
    """Test observational technique error model."""
    
    def test_technique_error_model_structure(self):
        """Test technique error model structure."""
        model = config.TECHNIQUE_ERROR_MODEL
        
        assert isinstance(model, dict)
        assert 'DEFAULT' in model
        
        # Check that all entries have required keys
        for technique, errors in model.items():
            assert isinstance(errors, dict)
            assert 'pa_error' in errors
            assert 'rho_error' in errors
            assert errors['pa_error'] > 0
            assert errors['rho_error'] > 0
    
    def test_technique_error_ranges(self):
        """Test that technique error ranges are reasonable."""
        model = config.TECHNIQUE_ERROR_MODEL
        
        # Space-based techniques should have smallest errors
        gaia_errors = model.get('G', model['DEFAULT'])
        micro_errors = model.get('M', model['DEFAULT'])
        
        # Gaia should be more precise than micrometry
        assert gaia_errors['pa_error'] <= micro_errors['pa_error']
        assert gaia_errors['rho_error'] <= micro_errors['rho_error']
        
        # All errors should be reasonable (not too large)
        for technique, errors in model.items():
            assert errors['pa_error'] <= 10.0  # Max 10 degrees
            assert errors['rho_error'] <= 1.0   # Max 1 arcsec


class TestWDSSConfiguration:
    """Test WDSS measurement column specifications."""
    
    def test_wdss_colspecs_structure(self):
        """Test WDSS column specifications structure."""
        colspecs = config.WDSS_MEASUREMENT_COLSPECS
        
        assert isinstance(colspecs, dict)
        expected_columns = [
            'wdss_id', 'pair', 'epoch', 'theta', 'theta_error',
            'rho', 'rho_error', 'mag1', 'mag2', 'reference', 'technique'
        ]
        
        for col in expected_columns:
            assert col in colspecs
            assert isinstance(colspecs[col], tuple)
            assert len(colspecs[col]) == 2
            assert colspecs[col][1] > colspecs[col][0]  # End > start


class TestCLIConfiguration:
    """Test CLI analysis configuration."""
    
    def test_cli_defaults(self):
        """Test CLI default values."""
        assert config.MIN_MEASUREMENTS_FOR_CHARACTERIZE >= 2
        assert config.DEFAULT_MIN_OBS >= 1
        assert config.DEFAULT_MAX_OBS > config.DEFAULT_MIN_OBS
        assert config.DEFAULT_CONCURRENT_REQUESTS > 0
        assert 0.0 < config.DEFAULT_GAIA_P_VALUE < 1.0
    
    def test_analysis_modes(self):
        """Test analysis mode configuration."""
        modes = config.ANALYSIS_MODES
        expected_modes = ['orbital', 'characterize', 'discovery']
        
        assert isinstance(modes, list)
        for mode in expected_modes:
            assert mode in modes
    
    def test_display_configuration(self):
        """Test display configuration."""
        assert config.TOP_RESULTS_DISPLAY_COUNT > 0
        assert config.DEFAULT_SORT_BY in ['v_total', 'opi', 'rmse', 'v_total_median']


class TestGUIConfiguration:
    """Test GUI configuration parameters."""
    
    def test_gui_defaults(self):
        """Test GUI default values."""
        assert config.GUI_DEFAULT_WIDTH > 0
        assert config.GUI_DEFAULT_HEIGHT > 0
        assert 0.0 <= config.DEFAULT_MIN_ALTITUDE_DEG <= 90.0
        assert config.DEFAULT_RA_WINDOW_HOURS > 0
        assert config.DEFAULT_LIGHT_POLLUTION_MAG > 0
    
    def test_gui_validation_ranges(self):
        """Test GUI parameter validation ranges."""
        assert config.MIN_ALTITUDE_DEG == 0.0
        assert config.MAX_ALTITUDE_DEG == 90.0
        assert config.MIN_RA_WINDOW_HOURS > 0
        assert config.MAX_RA_WINDOW_HOURS > config.MIN_RA_WINDOW_HOURS
        assert config.MIN_LIGHT_POLLUTION_MAG < config.MAX_LIGHT_POLLUTION_MAG


class TestStelleDoppieConfiguration:
    """Test Stelle Doppie search configuration."""
    
    def test_stelle_doppie_base_config(self):
        """Test Stelle Doppie base configuration."""
        assert isinstance(config.STELLE_DOPPIE_BASE_URL, str)
        assert config.STELLE_DOPPIE_BASE_URL.startswith("http")
        
        methods = config.STELLE_DOPPIE_SEARCH_METHODS
        assert isinstance(methods, dict)
        for method, value in methods.items():
            assert isinstance(value, int)
            assert value > 0
    
    def test_stelle_doppie_methods(self):
        """Test Stelle Doppie search methods."""
        methods = config.STELLE_DOPPIE_METHODS
        assert isinstance(methods, dict)
        
        # Check that method IDs are integers and descriptions are strings
        for method_id, description in methods.items():
            assert isinstance(method_id, int)
            assert isinstance(description, str)
            assert len(description) > 0
    
    def test_stelle_doppie_filters(self):
        """Test Stelle Doppie filter configuration."""
        filters = config.STELLE_DOPPIE_FILTERS
        assert isinstance(filters, dict)
        
        # Check required filter structure
        for filter_name, filter_config in filters.items():
            assert isinstance(filter_config, dict)
            required_keys = ['param_name', 'label', 'unit', 'data_type', 'available_methods']
            
            for key in required_keys:
                assert key in filter_config
            
            assert filter_config['data_type'] in ['numeric', 'string', 'integer']
            assert isinstance(filter_config['available_methods'], list)
    
    def test_default_search_options(self):
        """Test default search options."""
        options = config.DEFAULT_SEARCH_OPTIONS
        assert isinstance(options, dict)
        
        # All options should be boolean flags
        for option, value in options.items():
            assert isinstance(value, bool)


class TestExportConfiguration:
    """Test export format configuration."""
    
    def test_export_formats(self):
        """Test export format configuration."""
        formats = config.EXPORT_FORMATS
        assert isinstance(formats, dict)
        
        expected_formats = ['csv', 'json', 'fits', 'votable', 'latex']
        for fmt in expected_formats:
            assert fmt in formats
            
            format_config = formats[fmt]
            assert 'name' in format_config
            assert 'extension' in format_config
            assert 'mime_type' in format_config
            assert 'description' in format_config
            
            assert format_config['extension'].startswith('.')


class TestCatalogSourcesConfiguration:
    """Test catalog sources configuration."""
    
    def test_catalog_sources(self):
        """Test catalog sources configuration."""
        sources = config.CATALOG_SOURCES
        assert isinstance(sources, dict)
        
        expected_sources = ['stelle_doppie', 'wds', 'gaia', 'hipparcos']
        for source in expected_sources:
            assert source in sources
            
            source_config = sources[source]
            assert 'name' in source_config
            assert 'description' in source_config
            assert 'url' in source_config
            assert 'enabled' in source_config
            assert 'priority' in source_config
            
            assert isinstance(source_config['enabled'], bool)
            assert isinstance(source_config['priority'], int)
            assert source_config['priority'] > 0


class TestPlannerConfiguration:
    """Test planner module configuration."""
    
    def test_sky_quality_configuration(self):
        """Test sky quality map configuration."""
        assert config.DEFAULT_GRID_RESOLUTION_ARCMIN > 0
        assert config.FINE_GRID_RESOLUTION_ARCMIN > 0
        assert config.COARSE_GRID_RESOLUTION_ARCMIN > 0
        
        # Fine should be smaller than default, coarse should be larger
        assert config.FINE_GRID_RESOLUTION_ARCMIN < config.DEFAULT_GRID_RESOLUTION_ARCMIN
        assert config.COARSE_GRID_RESOLUTION_ARCMIN > config.DEFAULT_GRID_RESOLUTION_ARCMIN
    
    def test_extinction_coefficients(self):
        """Test atmospheric extinction coefficients."""
        # Test that all extinction coefficients are positive
        assert config.EXTINCTION_COEFFICIENT_U > 0
        assert config.EXTINCTION_COEFFICIENT_B > 0
        assert config.EXTINCTION_COEFFICIENT_V > 0
        assert config.EXTINCTION_COEFFICIENT_R > 0
        assert config.EXTINCTION_COEFFICIENT_I > 0
        assert config.DEFAULT_EXTINCTION_COEFFICIENT > 0
        
        # Test that U-band has highest extinction (bluer = more extinction)
        assert config.EXTINCTION_COEFFICIENT_U > config.EXTINCTION_COEFFICIENT_V
        assert config.EXTINCTION_COEFFICIENT_V > config.EXTINCTION_COEFFICIENT_I
    
    def test_sky_brightness_models(self):
        """Test sky brightness model values."""
        brightness_values = [
            config.PRISTINE_SKY_BRIGHTNESS_V_MAG_ARCSEC2,
            config.EXCELLENT_SKY_BRIGHTNESS_V_MAG_ARCSEC2,
            config.GOOD_SKY_BRIGHTNESS_V_MAG_ARCSEC2,
            config.MODERATE_SKY_BRIGHTNESS_V_MAG_ARCSEC2,
            config.POOR_SKY_BRIGHTNESS_V_MAG_ARCSEC2
        ]
        
        # All should be positive magnitudes
        for brightness in brightness_values:
            assert brightness > 0
            assert brightness < 30  # Reasonable upper limit
        
        # Should be in decreasing order (darker sites have higher mag/arcsec²)
        for i in range(len(brightness_values) - 1):
            assert brightness_values[i] > brightness_values[i + 1]
    
    def test_observational_limits(self):
        """Test observational limit configuration."""
        assert 0 <= config.MIN_OBSERVABLE_ALTITUDE_DEG <= 90
        assert 0 <= config.OPTIMAL_MIN_ALTITUDE_DEG <= 90
        assert 0 <= config.ZENITH_AVOIDANCE_ZONE_DEG <= 90
        
        # Optimal should be higher than minimum
        assert config.OPTIMAL_MIN_ALTITUDE_DEG >= config.MIN_OBSERVABLE_ALTITUDE_DEG
    
    def test_airmass_configuration(self):
        """Test airmass calculation configuration."""
        assert config.MAX_AIRMASS_FOR_PHOTOMETRY > 1.0
        assert config.MAX_AIRMASS_FOR_SPECTROSCOPY > 1.0
        assert config.AIRMASS_WARNING_THRESHOLD > 1.0
        
        # Spectroscopy should be more restrictive than photometry
        assert config.MAX_AIRMASS_FOR_SPECTROSCOPY <= config.MAX_AIRMASS_FOR_PHOTOMETRY


class TestPhysicsConfiguration:
    """Test physics module configuration."""
    
    def test_kepler_solver_configuration(self):
        """Test Kepler equation solver configuration."""
        assert config.DEFAULT_KEPLER_TOLERANCE > 0
        assert config.DEFAULT_KEPLER_TOLERANCE < 1e-6  # Should be small
        assert config.DEFAULT_KEPLER_MAX_ITERATIONS > 0
        assert 0 < config.HIGH_ECCENTRICITY_THRESHOLD < 1
        assert config.HIGH_E_COEFFICIENT > 1.0
        assert config.DANGEROUS_ECCENTRICITY_WARNING < 1.0
    
    def test_opi_configuration(self):
        """Test Observation Priority Index configuration."""
        assert config.OPI_DEVIATION_THRESHOLD_ARCSEC > 0
        assert config.OPI_INFINITE_THRESHOLD > 100  # Should be large
        assert config.MIN_POINTS_FOR_ROBUST_FIT >= 2
        assert isinstance(config.ROBUST_REGRESSION_RANDOM_STATE, int)
    
    def test_dynamics_validation_ranges(self):
        """Test dynamics validation ranges."""
        assert config.MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR > 0
        assert config.MIN_TIME_BASELINE_YEARS > 0
        assert config.MAX_CURVATURE_INDEX_ARCSEC > 0
        
        # Prediction limits should be reasonable
        assert config.MIN_PREDICTION_DATE_OFFSET_YEARS < 0
        assert config.MAX_PREDICTION_DATE_OFFSET_YEARS > 0
        assert config.MAX_EXTRAPOLATION_FACTOR > 1.0


class TestCLIDisplayConfiguration:
    """Test CLI display configuration."""
    
    def test_cli_display_parameters(self):
        """Test CLI display formatting parameters."""
        assert config.CLI_RESULTS_SEPARATOR_WIDTH > 0
        assert len(config.CLI_HEADER_CHAR) == 1
        assert len(config.CLI_SUBHEADER_CHAR) == 1
        
        # Precision values should be reasonable
        assert 0 <= config.CLI_OPI_PRECISION <= 10
        assert 0 <= config.CLI_RMSE_PRECISION <= 10
        assert 0 <= config.CLI_VELOCITY_PRECISION <= 10
        
        # Width values should be positive
        assert config.CLI_WDS_NAME_WIDTH > 0
        assert config.CLI_METRIC_LABEL_WIDTH > 0
        assert config.CLI_NUMERIC_FIELD_WIDTH > 0
    
    def test_cli_validation_ranges(self):
        """Test CLI validation configuration."""
        assert config.MIN_OBSERVATION_COUNT >= 1
        assert config.MAX_OBSERVATION_COUNT > config.MIN_OBSERVATION_COUNT
        assert config.MIN_CONCURRENT_REQUESTS >= 1
        assert config.MAX_CONCURRENT_REQUESTS > config.MIN_CONCURRENT_REQUESTS
        assert 0 < config.MIN_GAIA_P_VALUE < config.MAX_GAIA_P_VALUE <= 1.0


class TestDataSourceConfiguration:
    """Test data source configuration."""
    
    def test_data_sources_list(self):
        """Test available data sources."""
        sources = config.DATA_SOURCES
        assert isinstance(sources, list)
        expected_sources = ['web', 'local', 'gaia']
        
        for source in expected_sources:
            assert source in sources
    
    def test_gaia_configuration(self):
        """Test Gaia archive configuration."""
        assert isinstance(config.DEFAULT_GAIA_TABLE, str)
        assert config.DEFAULT_GAIA_TIMEOUT_SECONDS > 0
        
        # P-value thresholds should be properly ordered
        assert config.DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD < config.DEFAULT_PHYSICAL_P_VALUE_THRESHOLD
        assert config.DEFAULT_PHYSICAL_P_VALUE_THRESHOLD <= 1.0
    
    def test_gaia_quality_thresholds(self):
        """Test Gaia data quality thresholds."""
        assert config.MIN_PARALLAX_SIGNIFICANCE > 0
        assert config.MIN_PM_SIGNIFICANCE > 0
        assert config.GAIA_MAX_RUWE > 1.0  # RUWE > 1 indicates potential issues
        assert 0.0 <= config.GAIA_DEFAULT_CORRELATION_MISSING <= 1.0


class TestConfigurationConsistency:
    """Test internal consistency of configuration values."""
    
    def test_coordinate_range_consistency(self):
        """Test that coordinate ranges are consistent across modules."""
        # RA ranges should be consistent
        assert config.MIN_RA_DEG == 0.0
        assert config.MAX_RA_DEG == 360.0
        
        # Dec ranges should be consistent
        assert config.MIN_DEC_DEG == -90.0
        assert config.MAX_DEC_DEG == 90.0
    
    def test_error_threshold_consistency(self):
        """Test that error thresholds are internally consistent."""
        # Fallback errors should be smaller than validation thresholds
        orb6_fallbacks = config.ORB6_FALLBACK_ERRORS
        orb6_thresholds = config.ORB6_ERROR_VALIDATION_THRESHOLDS
        
        for key in orb6_fallbacks:
            if key in orb6_thresholds:
                assert orb6_fallbacks[key] <= orb6_thresholds[key]
    
    def test_gaia_threshold_consistency(self):
        """Test Gaia threshold consistency."""
        # Ambiguous threshold should be smaller than physical threshold
        assert config.DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD < config.DEFAULT_PHYSICAL_P_VALUE_THRESHOLD
        
        # Both should be valid probabilities
        assert 0 < config.DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD <= 1.0
        assert 0 < config.DEFAULT_PHYSICAL_P_VALUE_THRESHOLD <= 1.0
    
    def test_analysis_mode_consistency(self):
        """Test analysis mode configuration consistency."""
        modes = config.ANALYSIS_MODES
        available_modes = config.AVAILABLE_ANALYSIS_MODES
        
        # All modes should be available
        for mode in modes:
            assert mode in available_modes
        
        # Default mode should be in available modes
        assert config.DEFAULT_ANALYSIS_MODE in available_modes
        
        # Default sort keys should exist for all modes
        sort_keys = config.DEFAULT_SORT_KEYS
        for mode in available_modes:
            assert mode in sort_keys


if __name__ == "__main__":
    pytest.main([__file__])
