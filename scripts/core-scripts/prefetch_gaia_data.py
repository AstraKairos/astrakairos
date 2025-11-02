"""
    One of the issues I encountered with the original implementation was
    that requesting data from Gaia during the physicality analysis phase
    was extremely inefficient. This script pre-fetches all necessary Gaia
    data and stores it locally for later use.
"""