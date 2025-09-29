import pytest
import numpy as np
from unittest.mock import Mock, patch
from astrakairos.config import DEFAULT_GAIA_TABLE
from astrakairos.data.gaia_source import GaiaValidator


class FakeGaiaRow(dict):
    """Minimal Gaia record stand-in with dictionary semantics and colnames."""

    def __init__(self, data):
        super().__init__(data)
        self.colnames = list(data.keys())

class TestGaiaValidator:
    """Test GaiaValidator class."""
    
    @pytest.fixture
    def gaia_validator(self):
        return GaiaValidator(
            physical_p_value_threshold=0.05,
            ambiguous_p_value_threshold=0.001
        )
    
    def test_initialization_default(self):
        """Test GaiaValidator initialization with default values."""
        validator = GaiaValidator()
        
        assert validator.physical_threshold == 0.02
        assert validator.ambiguous_threshold == 0.0015
        assert validator.gaia_table == DEFAULT_GAIA_TABLE
    
    def test_initialization_custom_values(self):
        """Test GaiaValidator initialization with custom values."""
        validator = GaiaValidator(
            gaia_table='custom.gaia_source',
            physical_p_value_threshold=0.1,
            ambiguous_p_value_threshold=0.01
        )
        
        assert validator.physical_threshold == 0.1
        assert validator.ambiguous_threshold == 0.01
        assert validator.gaia_table == 'custom.gaia_source'
    
    def test_initialization_invalid_thresholds(self):
        """Test GaiaValidator initialization with invalid thresholds."""
        with pytest.raises(ValueError, match="physical_p_value_threshold \\(0.001\\) must be greater"):
            GaiaValidator(
                physical_p_value_threshold=0.001,
                ambiguous_p_value_threshold=0.01
            )
    
    @pytest.mark.asyncio
    async def test_validate_physicality_success(self, gaia_validator):
        """Test successful physicality validation using stored Gaia IDs."""
        from astrakairos.data.source import PhysicalityLabel, ValidationMethod

        primary_source_id = '123456789012345678'
        secondary_source_id = '987654321098765432'

        wds_summary = {
            'wds_id': '00000+0000',
            'component_pair': 'AB',
            'gaia_id_primary': f'DR3 {primary_source_id}',
            'gaia_id_secondary': f'ER3 {secondary_source_id}'
        }

        mock_primary = {'source_id': primary_source_id}
        mock_secondary = {'source_id': secondary_source_id}
        mock_gaia_results = [mock_primary, mock_secondary]

        mock_validation = {
            'label': PhysicalityLabel.LIKELY_PHYSICAL,
            'p_value': 0.12,
            'method': ValidationMethod.EXPERT_EL_BADRY,
            'expert_confidence': 0.85
        }

        with patch.object(gaia_validator, '_query_gaia_by_source_ids_async', return_value=mock_gaia_results):
            with patch.object(gaia_validator, '_validate_physicality_sync', return_value=(mock_validation, mock_primary, mock_secondary)) as mock_sync:
                with patch.object(gaia_validator, '_create_final_assessment', return_value={'label': PhysicalityLabel.LIKELY_PHYSICAL}) as mock_final:
                    result = await gaia_validator.validate_physicality(wds_summary)

        assert result['label'] == PhysicalityLabel.LIKELY_PHYSICAL
        mock_sync.assert_called_once_with(
            mock_gaia_results,
            wds_summary=wds_summary,
            gaia_source_ids={'A': primary_source_id, 'B': secondary_source_id}
        )
        mock_final.assert_called_once()
        args, kwargs = mock_final.call_args
        assert args == (mock_validation, mock_primary, mock_secondary)
        assert kwargs == {'search_radius_arcsec': None, 'direct_source_query': True}
    
    @pytest.mark.asyncio
    async def test_validate_physicality_requires_gaia_ids(self, gaia_validator):
        """Validation now requires Gaia IDs to proceed."""
        from astrakairos.exceptions import InsufficientAstrometricDataError

        wds_summary = {
            'wds_id': '00000+0000',
            'ra_deg': 15.0,
            'dec_deg': 45.0
        }

        with pytest.raises(InsufficientAstrometricDataError, match="Gaia source IDs required"):
            await gaia_validator.validate_physicality(wds_summary)

    def test_extract_gaia_source_ids_supported_formats(self, gaia_validator):
        """Gaia IDs are parsed from the supported WDS record formats for arbitrary component letters."""
        wds_summary = {
            'component_pair': 'AC',
            'name_primary': (
                '0556586-865740 A 2016 1 340 0.91 12.44 . ms -11.60 83.41 4.66 '
                'DR2 4613938814673524224 V 055658.67-865741.0'
            ),
            'name_secondary': (
                '0556586-865740 C 2016 340 0.91 12.93 . -13.70 83.77 4.54 '
                'ER3 4613938818970534144 055658.33-865740.2'
            )
        }

        result = gaia_validator._extract_gaia_source_ids(wds_summary)

        assert result == {
            'A': '4613938814673524224',
            'C': '4613938818970534144'
        }

    def test_extract_gaia_source_ids_single_non_numeric(self, gaia_validator):
        """Non-numeric identifiers do not yield a usable Gaia ID mapping."""
        wds_summary = {
            'component_pair': 'AB',
            'name_primary': (
                '0557038-131817 A 2015 1 280 51.56M 19.99 . -398.0 4.0 . '
                '[HrZ2020] SWB226740A U 055703.86-131817.3'
            )
        }

        assert gaia_validator._extract_gaia_source_ids(wds_summary) is None

    def test_extract_gaia_source_ids_from_json_mapping(self, gaia_validator):
        """Structured gaia_source_ids JSON is parsed directly into component mappings."""
        wds_summary = {
            'component_pair': 'AB',
            'gaia_source_ids': '{"A": "3573745362476642048", "B": "3573745362474959360", "component": "unused"}'
        }

        assert gaia_validator._extract_gaia_source_ids(wds_summary) == {
            'A': '3573745362476642048',
            'B': '3573745362474959360'
        }

    def test_extract_gaia_source_ids_from_primary_secondary_fields(self, gaia_validator):
        """Primary/secondary structured keys map using the declared component order."""
        wds_summary = {
            'component_pair': 'AC',
            'gaia_source_ids': '{"primary": "123456", "secondary": "654321"}'
        }

        assert gaia_validator._extract_gaia_source_ids(wds_summary) == {
            'A': '123456',
            'C': '654321'
        }
    
    def test_validate_physicality_sync_success(self, gaia_validator):
        """Test synchronous validation when Gaia IDs map directly to sources."""
        from astrakairos.data.source import PhysicalityLabel, ValidationMethod

        primary_id = '123456789012345678'
        secondary_id = '987654321098765432'
        gaia_results = [
            {'source_id': primary_id},
            {'source_id': secondary_id}
        ]
        wds_summary = {'component_pair': 'AB'}
        gaia_source_ids = {'A': primary_id, 'B': secondary_id}

        expert_result = {
            'label': PhysicalityLabel.LIKELY_PHYSICAL,
            'p_value': 0.08,
            'method': ValidationMethod.EXPERT_EL_BADRY,
            'expert_method': 'mock_expert',
            'expert_confidence': 0.9
        }

        with patch.object(gaia_validator, '_validate_astrometric_quality', return_value=True), \
             patch.object(gaia_validator, '_extract_expected_geometry', return_value=(None, None)), \
             patch.object(gaia_validator, '_verify_separation_consistency', return_value=True), \
             patch.object(gaia_validator, '_validate_with_expert_tree', return_value=expert_result), \
             patch.object(gaia_validator, '_calculate_statistical_consistency', return_value=None), \
             patch.object(gaia_validator, '_calculate_el_badry_metrics', return_value=None), \
             patch.object(gaia_validator, '_attach_expected_geometry') as mock_attach:

            result, primary_gaia, secondary_gaia = gaia_validator._validate_physicality_sync(
                gaia_results,
                wds_summary=wds_summary,
                gaia_source_ids=gaia_source_ids
            )

        assert result == expert_result
        assert primary_gaia is gaia_results[0]
        assert secondary_gaia is gaia_results[1]
        mock_attach.assert_called_once_with(expert_result, None, None)
    
    def test_validate_physicality_sync_not_enough_sources(self, gaia_validator):
        """Test synchronous physicality validation with insufficient sources."""
        from astrakairos.exceptions import InsufficientAstrometricDataError
        wds_summary = {'component_pair': 'AB'}
        primary_id = '123456789012345678'
        secondary_id = '987654321098765432'
        gaia_source_ids = {'A': primary_id, 'B': secondary_id}

        with pytest.raises(InsufficientAstrometricDataError, match="Only 1 Gaia sources returned"):
            gaia_validator._validate_physicality_sync(
                [{'source_id': primary_id}],
                wds_summary=wds_summary,
                gaia_source_ids=gaia_source_ids
            )

    def test_validate_physicality_sync_missing_components(self, gaia_validator):
        """Test failure when expected Gaia IDs are absent in query results."""
        from astrakairos.exceptions import InsufficientAstrometricDataError

        primary_id = '123456789012345678'
        secondary_id = '987654321098765432'
        gaia_results = [
            {'source_id': primary_id},
            {'source_id': '000000000000000000'}
        ]

        with patch.object(gaia_validator, '_validate_astrometric_quality', return_value=True):
            with pytest.raises(InsufficientAstrometricDataError, match="Gaia query did not return expected components"):
                gaia_validator._validate_physicality_sync(
                    gaia_results,
                    wds_summary={'component_pair': 'AB'},
                    gaia_source_ids={'A': primary_id, 'B': secondary_id}
                )

    @pytest.mark.asyncio
    async def test_get_parallax_data_requires_ids(self, gaia_validator):
        """Parallax retrieval requires Gaia source identifiers."""
        from astrakairos.exceptions import ParallaxDataUnavailableError

        with pytest.raises(ParallaxDataUnavailableError, match="Gaia source IDs required"):
            await gaia_validator.get_parallax_data({'wds_id': '00000+0000'})

    @pytest.mark.asyncio
    async def test_get_parallax_data_success(self, gaia_validator):
        """Parallax data is derived from direct Gaia ID queries."""
        primary_id = '123456789012345678'
        secondary_id = '987654321098765432'

        gaia_rows = [
            {
                'source_id': primary_id,
                'parallax': 10.0,
                'parallax_error': 0.5,
                'ruwe': 1.1,
                'phot_g_mean_mag': 12.0,
                'pmra': 3.0,
                'pmra_error': 0.2,
                'pmdec': -2.0,
                'pmdec_error': 0.2
            },
            {
                'source_id': secondary_id,
                'parallax': 9.5,
                'parallax_error': 0.4,
                'ruwe': 1.3,
                'phot_g_mean_mag': 12.5,
                'pmra': 2.8,
                'pmra_error': 0.3,
                'pmdec': -1.9,
                'pmdec_error': 0.3
            }
        ]

        wds_summary = {
            'component_pair': 'AB',
            'gaia_id_primary': f'DR3 {primary_id}',
            'gaia_id_secondary': f'DR3 {secondary_id}'
        }

        with patch.object(gaia_validator, '_query_gaia_by_source_ids_async', return_value=gaia_rows):
            result = await gaia_validator.get_parallax_data(wds_summary)

        assert result['gaia_source_id'] == primary_id
        assert pytest.approx(result['parallax']) == 10.0
        assert pytest.approx(result['parallax_error']) == 0.5
        assert result['source'] == 'gaia_dr3'
        assert result['significance'] > 0

    def test_validate_physicality_sync_fallback_to_statistical(self, gaia_validator):
        """When expert result is ambiguous, fall back to statistical evidence."""
        from astrakairos.data.source import PhysicalityLabel, ValidationMethod

        primary_id = '123456789012345678'
        secondary_id = '987654321098765432'
        gaia_results = [
            {'source_id': primary_id},
            {'source_id': secondary_id}
        ]
        gaia_source_ids = {'A': primary_id, 'B': secondary_id}
        wds_summary = {'components': 'AB'}

        expert_result = {
            'label': PhysicalityLabel.AMBIGUOUS,
            'p_value': None,
            'method': ValidationMethod.EXPERT_EL_BADRY,
            'expert_method': 'mock_expert',
            'expert_confidence': 0.4,
            'expert_reasoning': 'insufficient evidence'
        }
        statistical_result = {
            'label': PhysicalityLabel.LIKELY_PHYSICAL,
            'p_value': 0.02,
            'method': ValidationMethod.GAIA_3D_PARALLAX_PM
        }

        with patch.object(gaia_validator, '_validate_astrometric_quality', return_value=True), \
             patch.object(gaia_validator, '_extract_expected_geometry', return_value=(None, None)), \
             patch.object(gaia_validator, '_verify_separation_consistency', return_value=True), \
             patch.object(gaia_validator, '_validate_with_expert_tree', return_value=expert_result), \
             patch.object(gaia_validator, '_calculate_statistical_consistency', return_value=statistical_result), \
             patch.object(gaia_validator, '_calculate_el_badry_metrics', return_value=None), \
             patch.object(gaia_validator, '_attach_expected_geometry') as mock_attach:

            result, primary_gaia, secondary_gaia = gaia_validator._validate_physicality_sync(
                gaia_results,
                wds_summary=wds_summary,
                gaia_source_ids=gaia_source_ids
            )

        assert result['label'] == PhysicalityLabel.LIKELY_PHYSICAL
        assert result['method'] == ValidationMethod.GAIA_3D_PARALLAX_PM
        assert result['expert_confidence'] == expert_result['expert_confidence']
        assert result['expert_method'] == expert_result['expert_method']
        assert result['expert_reasoning'] == expert_result['expert_reasoning']
        assert primary_gaia is gaia_results[0]
        assert secondary_gaia is gaia_results[1]
        assert mock_attach.call_count == 2
        mock_attach.assert_any_call(expert_result, None, None)
        mock_attach.assert_any_call(statistical_result, None, None)
    
    def test_get_params_and_check_validity(self, gaia_validator):
        """Test parameter validity checking."""
        mock_star = Mock()
        mock_star.colnames = ['parallax', 'parallax_error', 'pmra']
        mock_star.__getitem__ = Mock(side_effect=lambda key: {
            'parallax': 5.0,
            'parallax_error': 0.1,
            'pmra': 10.0
        }[key])
        
        # Test valid parameters
        assert gaia_validator._get_params_and_check_validity(
            mock_star, ['parallax', 'parallax_error']
        ) == True
        
        # Test missing parameter
        assert gaia_validator._get_params_and_check_validity(
            mock_star, ['parallax', 'missing_param']
        ) == False
    
    def test_calculate_chi2_3d_success(self, gaia_validator):
        """Test successful 3D chi-squared calculation with full covariance."""
        star1 = FakeGaiaRow({
            'parallax': 5.0,
            'parallax_error': 0.1,
            'pmra': 10.0,
            'pmra_error': 0.2,
            'pmdec': -5.0,
            'pmdec_error': 0.15,
            'parallax_pmra_corr': 0.1,
            'parallax_pmdec_corr': 0.05,
            'pmra_pmdec_corr': 0.2,
            'ruwe': 1.1
        })
        star2 = FakeGaiaRow({
            'parallax': 4.8,
            'parallax_error': 0.12,
            'pmra': 9.8,
            'pmra_error': 0.18,
            'pmdec': -4.9,
            'pmdec_error': 0.14,
            'parallax_pmra_corr': 0.08,
            'parallax_pmdec_corr': 0.04,
            'pmra_pmdec_corr': 0.18,
            'ruwe': 1.05
        })

        chi2_val = gaia_validator._calculate_chi2_3d(star1, star2)

        assert chi2_val is not None
        assert isinstance(chi2_val, (float, np.floating))
        assert chi2_val >= 0

    def test_calculate_chi2_3d_missing_data(self, gaia_validator):
        """3D chi-squared should bail out when essential data are missing."""
        star1 = FakeGaiaRow({
            'parallax': 5.0,
            'pmra': None,  # Missing proper motion prevents computation
            'pmdec': -5.0,
            'parallax_error': 0.1,
            'pmra_error': 0.2,
            'pmdec_error': 0.15,
            'parallax_pmra_corr': 0.1,
            'parallax_pmdec_corr': 0.05,
            'pmra_pmdec_corr': 0.2
        })
        star2 = FakeGaiaRow({
            'parallax': 4.8,
            'pmra': 9.8,
            'pmdec': -4.9,
            'parallax_error': 0.12,
            'pmra_error': 0.18,
            'pmdec_error': 0.14,
            'parallax_pmra_corr': 0.08,
            'parallax_pmdec_corr': 0.04,
            'pmra_pmdec_corr': 0.18
        })

        result = gaia_validator._calculate_chi2_3d(star1, star2)
        assert result is None
    
    def test_validate_astrometric_quality(self, gaia_validator):
        """Test astrometric quality validation with representative Gaia rows."""
        good_source = FakeGaiaRow({
            'source_id': 12345,
            'ruwe': 1.2,
            'parallax': 10.0,
            'parallax_error': 2.0,
            'pmra': 5.0,
            'pmra_error': 1.0,
            'pmdec': 3.0,
            'pmdec_error': 1.0
        })
        assert gaia_validator._validate_astrometric_quality(good_source) is True

        high_ruwe_source = FakeGaiaRow({
            'source_id': 12346,
            'ruwe': 15.0,
            'parallax': 10.0,
            'parallax_error': 2.0,
            'pmra': 5.0,
            'pmra_error': 1.0,
            'pmdec': 3.0,
            'pmdec_error': 1.0
        })
        assert gaia_validator._validate_astrometric_quality(high_ruwe_source) is False

        no_ruwe_source = FakeGaiaRow({
            'source_id': 12347,
            'parallax': 10.0,
            'parallax_error': 2.0
        })
        assert gaia_validator._validate_astrometric_quality(no_ruwe_source) is True