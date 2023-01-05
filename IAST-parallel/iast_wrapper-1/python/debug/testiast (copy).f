      Program Testiast
      Implicit None

      Include 'commons.inc'
C     See backup folder 
      Integer J,I,Nterm_max,Carrier_gas
      
      Double Precision P,Yi(Maxcomp),Xi1(Maxcomp)
     $     ,Ni(Maxcomp),Molfrac(Maxcomp,Maxterm)
C     Note to self: program written in Fortran 2003. 
C     An Inert Component Should Have
C     Nterm=1
C     Langmuir=True
C     Ki=0
C     Nimax Not Equal To Zero
C     Nterm_max means max no. of sides in pure-component isotherms. 
      
      Nterm_max = 2
C     Start for Python
      Ncomp = 2
      Langmuir(1, 2) = .True. 
      Langmuir(1, 1) = .True. 
      Pow(2, 2) = 1.0d0 
      Pow(2, 1) = 1.0d0 
      Langmuir(1, 2) = .True. 
      Langmuir(1, 1) = .True. 
      Pow(1, 2) = 1.0d0 
      Pow(1, 1) = 1.0d0 
      Nimax(2, 2) = 0.815592d0 
      Nimax(2, 1) = 0.680083d0 
      Ki(2, 2) = 0.00000000305d0 
      Ki(2, 1) = 0.01566805729d0 
      Nimax(1, 2) = 0.831311d0 
      Nimax(1, 1) = 0.675110d0 
      Ki(1, 2) = 0.00000000374d0 
      Ki(1, 1) = 0.02518823565d0 
      Yi(1) = 1.00d0      
      Yi(2) = 0.00d0      
C     End for Python


      Carrier_gas = 1       
      P   = 1.0d4
      Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)

      write(6,*) 'Ni(1)   ','Ni(2)   '
      write(6,'(2e20.10)') Ni(1),Ni(2)
      write(25,'(A)')   "     Pressure (Pa)     C7-500 (mol/kg)        3mC6-500 (mol/kg)"      
C      stop

      Do J = 0, 8
         If(J.Lt.8) Then
            Do I = 1, 100, 2
               P = Dble(I*10**J)
               Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
               Write(25,'(5e20.10)') P,Ni(1),Ni(2)
            Enddo
         Else
            P = Dble(1.0*10**J)
            Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
            Write(25,'(5e20.10)') P,Ni(1),Ni(2)
         Endif
      Enddo
      
      
      Stop
      End
      
