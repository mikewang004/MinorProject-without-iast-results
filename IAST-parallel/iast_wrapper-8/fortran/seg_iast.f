      Subroutine Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
      Implicit None
      Include 'commons.inc'
      
C     solve the iast equations for a Langmuir isotherms
      
C     Yi  = gas phase molefraction
C     Xi1, Molfrac  = adsorbed phase molefraction
C     Pi0 = upper boundary integral spreading pressure component i
C     Pi  = spreading pressure
C     Ni1  = number of adsorbed molecules of component i
C     Nterm_max = maximum number of terms in the pure component isotherm

      Double Precision Ni(Maxcomp),Ni1(Maxcomp),Yi(Maxcomp),P
     $                 ,Qeq(Maxcomp,Maxterm),Xi1(Maxcomp)
     $                 ,Pi0(Maxcomp),Pi,Molfrac(Maxcomp,Maxterm)
      Integer I,term,Nterm_max,Carrier_gas

      Do term = 1,Nterm_max
         Call Iast_per_site(Ni1,Xi1,Yi,P,Pi0,Pi,term,Carrier_gas)
         Do I=1,Ncomp
            Qeq(I,term) = Ni1(I)
            Molfrac(I,term) = Xi1(I)
         Enddo         
      Enddo

      Do I=1,Ncomp
         Ni(I) = 0.0d0
         Do term=1,Nterm_max
            Ni(I) = Ni(I)+Qeq(I,term)            
         Enddo
      Enddo

      Return
      End

C     *********************************************************************
      Subroutine Iast_per_site(Ni1,Xi1,Yi,P,Pi0,Pi,term,Carrier_gas)
      Implicit None

C     solve the iast equations for a Langmuir isotherms per site
      
C     Yi  = gas phase molefraction
C     Xi1  = adsorbed phase molefraction
C     Pi0 = upper boundary integral spreading pressure component i
C     Pi  = spreading pressure
C     Ni1  = number of adsorbed molecules of component i
C     Nterm_max = maximum number of terms in the pure component isotherm
      
      Include 'commons.inc'
      
      Double Precision Yi(Maxcomp),P,Xi1(Maxcomp),Pi0(Maxcomp)
     $     ,Pi,Ni1(Maxcomp),Pi1,Pi2,Pim
     $     ,Sum1,Sum2,Summ,Tiny
     $     ,SpreadingPressure,InvP,IsothermPure,Error

      Integer I,term,Carrier_gas

      Parameter (Tiny = 1.0d-15)
      
      If(Dabs(Yi(Carrier_gas)-1.0d0).Lt.Tiny) Then

         Pi = 0.0d0

         Do I=1,Ncomp
            Xi1(I)  = 0.0d0
            Pi0(I) = 0.0d0
            Ni1(I) = 0.0d0
         Enddo                             
         
         Return
      Endif
            
C     Get Initial Estimate Of The Spreading Pressure
C     we assume here that Xi equals Yi so Pi0 equals P
      
      Pi1 = 0.0d0

      Do I=1,Ncomp            
         Pi1 = Pi1 + Yi(I)*spreadingpressure(I,term,P)
      Enddo        
         
         
C     For This Initial Estimate Pi1 Compute The Sum Of Molefractions
      
C     Initialize The Bisection Algorithm


      If(Pi1.Lt.Tiny) Then
C     Nothing Is Adsorbing
C     Either because P equals zero, or all the Ki values are zero
C     so we immediately know the solution
         Pi = 0.0d0
         
         Do I=1,Ncomp
            Xi1(I)  = 0.0d0
            Pi0(I) = 0.0d0
            Ni1(I) = 0.0d0
         Enddo          
         Return
      Endif
            
      Sum1 = 0.0d0
      
      Do I=1,Ncomp
         Sum1 = Sum1 + Yi(I)*P*Invp(I,term,Pi1)
      Enddo
      
      If(Sum1.Gt.1.0d0) Then

C     make an estimate for the spreading pressure where the sum of molefractions
C     is smaller than 1
         
         Pi2 = Pi1
 1       Continue
         Pi2 = Pi2*2.0d0

         Sum2 = 0.0d0

         Do I=1,Ncomp
            Sum2 = Sum2 + Yi(I)*P*Invp(I,term,Pi2)
         Enddo
         
         If(Sum2.Gt.1.0d0) Goto 1
         
      Else
         Sum2 = Sum1
         Pi2  = Pi1

C     make an initial estimate for the spreading pressure when the sum of the molefractions
C     is larger than 1
         
 2       Continue
         Pi1 = Pi1*0.5d0

         Sum1 = 0.0d0

         Do I=1,Ncomp
            Sum1 = Sum1 + Yi(I)*P*Invp(I,term,Pi1)
         Enddo

         If(Sum1.Lt.1.0d0) Goto 2
         
      Endif
                            
C     Bisection
  
 3    Continue
      Pim  = 0.5d0*(Pi1+Pi2)
      Summ = 0.0d0

      Do I=1,Ncomp
         Summ = Summ + Yi(I)*P*Invp(I,term,Pim)
      Enddo
      
      If(Summ.Gt.1.0d0) Then
         Pi1 = Pim
      Else
         Pi2 = Pim
      Endif
         
C     test for convergence of the bisection
c     we need to test carefully if this converges to machine precision
         
      Error = Dabs(Pi1-Pi2)/Dabs(Pi1+Pi2)
      
      If(Error.Gt.Tiny) Goto 3

      If(Dabs(Summ-1.0d0).Gt.0.01d0) Then
        write(*,*) 'Error: mol-fraction not unity', Summ
        Stop
      Endif
      
      Pi = 0.5d0*(Pi1+Pi2)    
      Summ = 0.0d0

C     calculate molefractions in adsorbed phase and total loading
         
      Do I=1,Ncomp
         Xi1(I) = Yi(I)*P*Invp(I,term,Pi)

         If(Xi1(I).Gt.Tiny) Then
            Pi0(I) = Yi(I)*P/Xi1(I)

            If(IsothermPure(I,term,Pi0(I)).Gt.Tiny) Then
               Summ = Summ +
     &              Xi1(I)/IsothermPure(I,term,Pi0(I))
            Else
               Summ = Summ
            Endif               

         Else
            Pi0(I) = 0.0d0
            Xi1(I)  = 0.0d0
         Endif
      Enddo
      
C     calculate loading for all of the components
         
      If (Summ.Eq.0.0d0) Then
         Do I=1,Ncomp
            Ni1(I) = 0.0d0
         Enddo
      Else
         Summ = 1.0d0/Summ
         Do I=1,Ncomp
            Ni1(I) = Summ*Xi1(I)
         Enddo
      Endif    
      
      Return
      End
      
