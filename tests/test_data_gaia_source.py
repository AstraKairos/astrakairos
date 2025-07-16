import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from astrakairos.data.gaia_source import GaiaValidator
# --- Test GaiaValidator Class ---

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
        
        assert validator.physical_threshold == 0.05
        assert validator.ambiguous_threshold == 0.001
        assert validator.default_search_radius_arcsec == 10.0
        assert validator.mag_limit == 18.0
    
    def test_initialization_custom_values(self):
        """Test GaiaValidator initialization with custom values."""
        validator = GaiaValidator(
            physical_p_value_threshold=0.1,
            ambiguous_p_value_threshold=0.01,
            default_search_radius_arcsec=15.0,
            mag_limit=16.0
        )
        
        assert validator.physical_threshold == 0.1
        assert validator.ambiguous_threshold == 0.01
        assert validator.default_search_radius_arcsec == 15.0
        assert validator.mag_limit == 16.0
    
    def test_initialization_legacy_parameter(self):
        """Test GaiaValidator initialization rejects legacy parameter."""
        with pytest.raises(TypeError, match="unexpected keyword argument 'p_value_threshold'"):
            GaiaValidator(p_value_threshold=0.1)
    
    def test_initialization_invalid_thresholds(self):
        """Test GaiaValidator initialization with invalid thresholds."""
        with pytest.raises(ValueError, match="physical_p_value_threshold \\(0.001\\) must be greater"):
            GaiaValidator(
                physical_p_value_threshold=0.001,
                ambiguous_p_value_threshold=0.01
            )
    
    @pytest.mark.asyncio
    async def test_validate_physicality_success(self, gaia_validator):
        """Test successful physicality validation."""
        wds_summary = {
            'wds_id': '00000+0000',
            'ra_deg': 15.0,
            'dec_deg': 45.0,
            'mag_pri': 8.5,
            'mag_sec': 9.2
        }
        
        # Mock astropy table-like structure
        from astropy.table import Table
        
        # Create mock table with required columns
        mock_table = Table({
            'source_id': [123456789, 987654321],
            'ra': [15.0, 15.001],
            'dec': [45.0, 45.001],
            'parallax': [10.0, 10.2],
            'parallax_error': [0.1, 0.12],
            'pmra': [5.0, 5.1],
            'pmra_error': [0.05, 0.06],
            'pmdec': [-3.0, -2.9],
            'pmdec_error': [0.04, 0.05],
            'ruwe': [1.1, 1.2],
            'phot_g_mean_mag': [8.5, 9.2],
            'parallax_pmra_corr': [0.1, 0.1],
            'parallax_pmdec_corr': [0.1, 0.1],
            'pmra_pmdec_corr': [0.1, 0.1]
        })
        
        # Mock the query to return individual rows
        async def mock_query_star_data(wds_id, position, radius=5.0):
            # Return each row in sequence
            if not hasattr(mock_query_star_data, 'call_count'):
                mock_query_star_data.call_count = 0
            if mock_query_star_data.call_count < len(mock_table):
                row = mock_table[mock_query_star_data.call_count]
                mock_query_star_data.call_count += 1
                return row
            return None
        
        with patch.object(gaia_validator, 'query_star_data', side_effect=mock_query_star_data):
            result = await gaia_validator.validate_physicality(wds_summary)
            
            # The result should be a PhysicalityAssessment with proper structure
            assert 'label' in result
            assert 'confidence' in result
            assert 'p_value' in result
            assert 'method' in result
            assert 'validation_date' in result
            assert 'search_radius_arcsec' in result
            assert 'significance_thresholds' in result
            assert 'retry_attempts' in result
            
            # Check that the label is a valid enum value
            from astrakairos.data.source import PhysicalityLabel
            assert isinstance(result['label'], PhysicalityLabel)
            
            # Check that method is a valid enum value
            from astrakairos.data.source import ValidationMethod
            assert isinstance(result['method'], ValidationMethod)
    
    @pytest.mark.asyncio
    async def test_validate_physicality_missing_coordinates(self, gaia_validator):
        """Test physicality validation with missing coordinates."""
        wds_summary = {
            'wds_id': '00000+0000',
            'mag_pri': 8.5,
            'mag_sec': 9.2
            # Missing ra_deg and dec_deg
        }
        
        result = await gaia_validator.validate_physicality(wds_summary)
        
        # The result should be a PhysicalityAssessment with Unknown label
        from astrakairos.data.source import PhysicalityLabel
        assert result['label'] == PhysicalityLabel.UNKNOWN
        assert result['p_value'] is None
        assert 'method' in result
        assert 'validation_date' in result
    
    def test_validate_physicality_sync_success(self, gaia_validator):
        """Test synchronous physicality validation."""
        with patch.object(gaia_validator, '_query_gaia_for_pair_sync') as mock_query:
            with patch.object(gaia_validator, '_identify_components_by_mag') as mock_identify:
                with patch.object(gaia_validator, '_calculate_chi2_3d') as mock_chi2:
                    
                    # Mock Gaia query results
                    mock_star1 = Mock()
                    mock_star2 = Mock()
                    mock_query.return_value = [mock_star1, mock_star2]
                    
                    # Mock component identification
                    mock_identify.return_value = (mock_star1, mock_star2)
                    
                    # Mock chi-squared calculation
                    mock_chi2.return_value = (5.0, 3)  # chi2 value, dof
                    
                    result = gaia_validator._validate_physicality_sync(
                        [mock_star1, mock_star2],  # gaia_results
                        (8.5, 9.2)  # wds_mags
                    )
                    
                    assert result['label'] == 'Likely Physical'  # p > 0.05
                    assert result['p_value'] is not None
                    assert result['test_used'] == '3D (plx+pm)'
    
    def test_validate_physicality_sync_not_enough_sources(self, gaia_validator):
        """Test synchronous physicality validation with insufficient sources."""
        with patch.object(gaia_validator, '_query_gaia_for_pair_sync') as mock_query:
            mock_query.return_value = [Mock()]  # Only one source
            
            result = gaia_validator._validate_physicality_sync(
                [Mock()],  # Only one source
                (8.5, 9.2)  # wds_mags
            )
            
            assert result['label'] == 'Unknown'
            assert result['p_value'] is None
            assert result['test_used'] == 'Not enough Gaia sources'
    
    def test_validate_physicality_sync_component_matching_failed(self, gaia_validator):
        """Test synchronous physicality validation with component matching failure."""
        with patch.object(gaia_validator, '_query_gaia_for_pair_sync') as mock_query:
            with patch.object(gaia_validator, '_identify_components_by_mag') as mock_identify:
                
                mock_query.return_value = [Mock(), Mock()]
                mock_identify.return_value = (None, None)
                
                result = gaia_validator._validate_physicality_sync(
                    [Mock(), Mock()],  # gaia_results
                    (8.5, 9.2)  # wds_mags
                )
                
                assert result['label'] == 'Ambiguous'
                assert result['p_value'] is None
                assert result['test_used'] == 'Component matching failed'
    
    def test_validate_physicality_sync_fallback_tests(self, gaia_validator):
        """Test synchronous physicality validation with fallback to 2D and 1D tests."""
        with patch.object(gaia_validator, '_query_gaia_for_pair_sync') as mock_query:
            with patch.object(gaia_validator, '_identify_components_by_mag') as mock_identify:
                with patch.object(gaia_validator, '_calculate_chi2_3d') as mock_chi2_3d:
                    with patch.object(gaia_validator, '_calculate_chi2_2d_pm') as mock_chi2_2d:
                        
                        mock_star1 = Mock()
                        mock_star2 = Mock()
                        mock_query.return_value = [mock_star1, mock_star2]
                        mock_identify.return_value = (mock_star1, mock_star2)
                        
                        # 3D test fails, 2D test succeeds
                        mock_chi2_3d.return_value = None
                        mock_chi2_2d.return_value = (3.0, 2)
                        
                        result = gaia_validator._validate_physicality_sync(
                            [mock_star1, mock_star2],  # gaia_results
                            (8.5, 9.2)  # wds_mags
                        )
                        
                        assert result['test_used'] == '2D (pm_only)'
                        assert result['p_value'] is not None
    
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
        """Test successful 3D chi-squared calculation."""
        mock_star1 = Mock()
        mock_star1.colnames = ['parallax', 'parallax_error', 'pmra', 'pmra_error', 
                              'pmdec', 'pmdec_error', 'parallax_pmra_corr', 
                              'parallax_pmdec_corr', 'pmra_pmdec_corr']
        mock_star1.__getitem__ = Mock(side_effect=lambda key: {
            'parallax': 5.0, 'parallax_error': 0.1,
            'pmra': 10.0, 'pmra_error': 0.2,
            'pmdec': -5.0, 'pmdec_error': 0.15,
            'parallax_pmra_corr': 0.1, 'parallax_pmdec_corr': 0.05,
            'pmra_pmdec_corr': 0.2
        }[key])
        # Add get method for correlation coefficients
        mock_star1.get = Mock(side_effect=lambda key, default=None: {
            'parallax_pmra_corr': 0.1, 'parallax_pmdec_corr': 0.05, 'pmra_pmdec_corr': 0.2
        }.get(key, default))
        
        mock_star2 = Mock()
        mock_star2.colnames = mock_star1.colnames
        mock_star2.__getitem__ = Mock(side_effect=lambda key: {
            'parallax': 4.8, 'parallax_error': 0.12,
            'pmra': 9.8, 'pmra_error': 0.18,
            'pmdec': -4.9, 'pmdec_error': 0.14,
            'parallax_pmra_corr': 0.08, 'parallax_pmdec_corr': 0.04,
            'pmra_pmdec_corr': 0.18
        }[key])
        # Add get method for correlation coefficients
        mock_star2.get = Mock(side_effect=lambda key, default=None: {
            'parallax_pmra_corr': 0.08, 'parallax_pmdec_corr': 0.04, 'pmra_pmdec_corr': 0.18
        }.get(key, default))
        
        with patch.object(gaia_validator, '_get_params_and_check_validity', return_value=True):
            result = gaia_validator._calculate_chi2_3d(mock_star1, mock_star2)
            
            assert result is not None
            chi2_val, dof = result
            assert isinstance(chi2_val, (float, np.floating))
            assert dof == 3
            assert chi2_val >= 0
    
    def test_calculate_chi2_3d_missing_data(self, gaia_validator):
        """Test 3D chi-squared calculation with missing data."""
        mock_star1 = Mock()
        mock_star2 = Mock()
        
        with patch.object(gaia_validator, '_get_params_and_check_validity', return_value=False):
            result = gaia_validator._calculate_chi2_3d(mock_star1, mock_star2)
            assert result is None
    
    def test_identify_components_by_mag_with_magnitudes(self, gaia_validator):
        """Test component identification with WDS magnitudes."""
        mock_source1 = Mock()
        mock_source1.__getitem__ = Mock(return_value=8.5)
        mock_source1.get = Mock(return_value='source1')
        
        mock_source2 = Mock()
        mock_source2.__getitem__ = Mock(return_value=9.2)
        mock_source2.get = Mock(return_value='source2')
        
        # Mock the source_id differently for each source
        type(mock_source1).source_id = 'source1'
        type(mock_source2).source_id = 'source2'
        
        gaia_results = [mock_source1, mock_source2]
        wds_mags = (8.5, 9.2)
        
        primary, secondary = gaia_validator._identify_components_by_mag(gaia_results, wds_mags)
        
        assert primary == mock_source1
        assert secondary == mock_source2
    
    def test_identify_components_by_mag_without_magnitudes(self, gaia_validator):
        """Test component identification without WDS magnitudes."""
        mock_source1 = Mock()
        mock_source2 = Mock()
        
        gaia_results = [mock_source1, mock_source2]
        wds_mags = (None, None)
        
        primary, secondary = gaia_validator._identify_components_by_mag(gaia_results, wds_mags)
        
        assert primary == mock_source1
        assert secondary == mock_source2
    
    def test_query_gaia_for_pair_success(self, gaia_validator):
        """Test successful Gaia query."""
        # Create mock results that support both len() and slicing
        mock_results = [Mock(), Mock()]  # List of 2 mock sources
        
        with patch('astrakairos.data.gaia_source.Gaia') as mock_gaia:
            mock_job = Mock()
            mock_job.get_results.return_value = mock_results
            mock_gaia.launch_job.return_value = mock_job
            
            result = gaia_validator._query_gaia_for_pair_sync(15.0, 45.0, 10.0)
            
            # Should return the results (limited to max_sources)
            assert result == mock_results
            mock_gaia.launch_job.assert_called_once()
    
    def test_query_gaia_for_pair_exception(self, gaia_validator):
        """Test Gaia query with exception."""
        with patch('astrakairos.data.gaia_source.Gaia') as mock_gaia:
            mock_gaia.launch_job.side_effect = Exception("Network error")
            
            # The sync method now propagates exceptions instead of returning None
            with pytest.raises(Exception, match="Network error"):
                gaia_validator._query_gaia_for_pair_sync(15.0, 45.0, 10.0)
    
    def test_validate_astrometric_quality(self, gaia_validator):
        """Test astrometric quality validation."""
        # Good quality source
        good_source = Mock()
        good_source.get.side_effect = lambda key: {
            'source_id': 12345,
            'ruwe': 1.2,
            'parallax': 10.0,
            'parallax_error': 2.0,
            'pmra': 5.0,
            'pmra_error': 1.0,
            'pmdec': 3.0,
            'pmdec_error': 1.0
        }.get(key)
        good_source.colnames = ['source_id', 'ruwe', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error']
        good_source.__getitem__ = lambda self, key: {
            'source_id': 12345,
            'ruwe': 1.2,
            'parallax': 10.0,
            'parallax_error': 2.0,
            'pmra': 5.0,
            'pmra_error': 1.0,
            'pmdec': 3.0,
            'pmdec_error': 1.0
        }[key]
        
        assert gaia_validator._validate_astrometric_quality(good_source) == True
        
        # Poor RUWE source
        poor_ruwe_source = Mock()
        poor_ruwe_source.get.side_effect = lambda key: {
            'source_id': 12346,
            'ruwe': 2.0  # Too high
        }.get(key)
        poor_ruwe_source.colnames = ['source_id', 'ruwe']
        poor_ruwe_source.__getitem__ = lambda self, key: {
            'source_id': 12346,
            'ruwe': 2.0
        }[key]
        
        assert gaia_validator._validate_astrometric_quality(poor_ruwe_source) == False
        
        # Missing RUWE should still pass
        no_ruwe_source = Mock()
        no_ruwe_source.get.side_effect = lambda key: {
            'source_id': 12347,
            'parallax': 10.0,
            'parallax_error': 2.0
        }.get(key)
        no_ruwe_source.colnames = ['source_id', 'parallax', 'parallax_error']
        no_ruwe_source.__getitem__ = lambda self, key: {
            'source_id': 12347,
            'parallax': 10.0,
            'parallax_error': 2.0
        }[key]
        
        assert gaia_validator._validate_astrometric_quality(no_ruwe_source) == True