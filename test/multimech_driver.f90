! Selects a bundled mechanism and prints reference quantities, so the harness can
! compare them against the dedicated single-mechanism module's output.
program multimech_driver
    use m_thermochem
    implicit none
    real(dp) :: h(num_species_max), cp(num_species_max), c(num_species_max)
    real(dp) :: y(num_species_max), omega(num_species_max)
    real(dp) :: mw, t, rho
    integer :: status, i
    character(len=32) :: name
    call get_command_argument(1, name)
    call set_mechanism(trim(name), status)
    if (status /= 0) then
        print *, "UNKNOWN MECHANISM"
        stop 1
    end if
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
end program multimech_driver
