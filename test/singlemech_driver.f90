! Single-mechanism reference driver: prints the same quantities as multimech_driver.

program singlemech_driver
    use m_thermochem
    implicit none
    real(dp) :: h(num_species), cp(num_species), c(num_species)
    real(dp) :: y(num_species), omega(num_species)
    real(dp) :: mw, t, rho
    integer :: i
    t = 1200.0_dp
    rho = 0.2_dp
    y = 0.0_dp
    do i = 1, num_species
        y(i) = 1.0_dp/real(num_species, dp)
    end do
    call get_species_enthalpies_rt(t, h)
    call get_species_specific_heats_r(t, cp)
    call get_mixture_molecular_weight(y, mw)
    call get_concentrations(rho, y, c)
    call get_net_production_rates(rho, t, y, omega)
    do i = 1, num_species
        print '(5E24.15)', h(i), cp(i), c(i), omega(i), mw
    end do
end program singlemech_driver
