  [Mesh]
   [file]
   type = FileMeshGenerator
   file = input_example.e
   []

   [bound]
     type=SideSetsAroundSubdomainGenerator
     new_boundary = bound
     input  = file
     block =  9
   []
  []

 [GlobalParams]
   displacements = 'disp_x disp_y '
   youngs_modulus = 43
   poissons_ratio = 0.3
 []

 [Variables]
   [./disp_x]
   [../]
   [./disp_y]
   [../]
 []

 [AuxVariables]

   [./vonMises]
     family = MONOMIAL
     order = FIRST
   [../]
   [./conc]
     initial_from_file_var = concentration
   [../]
   [./comp]
     initial_from_file_var = composition
   [../]

   [./sxx]
     family = MONOMIAL
     order = FIRST
   [../]
   [./syy]
     family = MONOMIAL
     order = FIRST
   [../]
    [./sxy]
      family = MONOMIAL
      order = FIRST
    [../]
    [./exx]
      family = MONOMIAL
      order = FIRST
    [../]
    [./eyy]
      family = MONOMIAL
      order = FIRST
    [../]
     [./exy]
       family = MONOMIAL
       order = FIRST
     [../]

 []
 [AuxKernels]
   [./vonMises]
     type = RankTwoScalarAux
     rank_two_tensor = stress
     variable = vonMises
     scalar_type = VonMisesStress
   [../]
   [./sxx]
     type = RankTwoAux
     rank_two_tensor = stress
     variable = sxx
     index_i = 0
     index_j = 0
   [../]
   [./syy]
     type = RankTwoAux
     rank_two_tensor = stress
     variable = syy
     index_i = 1
     index_j = 1
   [../]
    [./sxy]
      type = RankTwoAux
      rank_two_tensor = stress
      variable = sxy
      index_i = 0
      index_j = 1
    [../]
    [./exx]
      type = RankTwoAux
      rank_two_tensor = total_strain
      variable = exx
      index_i = 0
      index_j = 0
    [../]
    [./eyy]
      type = RankTwoAux
      rank_two_tensor = total_strain
      variable = eyy
      index_i = 1
      index_j = 1
    [../]
     [./exy]
       type = RankTwoAux
       rank_two_tensor = total_strain
       variable = exy
       index_i = 0
       index_j = 1
     [../]

  []

 [Kernels]

   [./TensorMechanics]
     displacements = 'disp_x disp_y'
     strain = SMALL
   [../]
 []


 [Materials]
   [./Cijkl]
     type = ComputeIsotropicElasticityTensor
   [../]
   [./strain]
       type = ComputeSmallStrain
       eigenstrain_names = 'eigen_true'
     [../]
     [./stress]
       type = ComputeLinearElasticStress
     [../]
    [./prefactor]
       type = DerivativeParsedMaterial
       args = conc
       f_name = prefactor
       constant_names = 'epsilon0'
       constant_expressions = '2.415e-6'   #2.415e-6 ~ mol /3
       function = '(conc/3) * epsilon0'
     [../]
     [./eigen_strain_tensor]
       type = ComputeVariableEigenstrain
       args = conc
       eigen_base = '1 1 0 0 0 0'
       prefactor = prefactor
       eigenstrain_name = eigen_true
     [../]
 []

 [BCs]
   [./leftx]
     type = PresetBC
     variable = disp_x
     boundary = bound
     value = 0
   [../]
   [./lefty]
     type = PresetBC
     variable = disp_y
     boundary = bound
     value = 0
   [../]
 []

 [Preconditioning]
   [./smp]
     type = SMP
     full = true
   [../]
 []


 [Executioner]
   type = Steady
   automatic_scaling = true
   compute_scaling_once= false
   l_abs_tol = 1e-7
   nl_abs_tol = 1e-7
 []

 [Outputs]
   execute_on = 'timestep_end'
   exodus = true
   print_linear_residuals = true

 []
