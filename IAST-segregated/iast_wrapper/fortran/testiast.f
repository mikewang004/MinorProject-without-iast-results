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
C     Start for Python1
      Ncomp = 5 
      Yi(5) =  0.05d0 
      Langmuir(5, 2) = .True. 
      Langmuir(5, 1) = .True. 
      Pow(5, 2) = 1.0d0 
      Pow(5, 1) = 1.0d0 
      Nimax(5, 2) = 0.004168d0 
      Nimax(5, 1) = 0.737001d0 
      Ki(5, 2) = 2938.80509000000d0 
      Ki(5, 1) = 0.00026896701d0 
      Yi(4) =  0.10d0 
      Langmuir(4, 2) = .True. 
      Langmuir(4, 1) = .True. 
      Pow(4, 2) = 1.0d0 
      Pow(4, 1) = 1.0d0 
      Nimax(4, 2) = 0.667098d0 
      Nimax(4, 1) = 1.247270d0 
      Ki(4, 2) = 0.00015622902d0 
      Ki(4, 1) = 0.00000000142d0 
      Yi(3) =  0.15d0 
      Langmuir(3, 2) = .True. 
      Langmuir(3, 1) = .True. 
      Pow(3, 2) = 1.0d0 
      Pow(3, 1) = 1.0d0 
      Nimax(3, 2) = 0.330950d0 
      Nimax(3, 1) = 0.363662d0 
      Ki(3, 2) = 0.00018168848d0 
      Ki(3, 1) = 0.00018169289d0 
      Yi(2) =  0.20d0 
      Langmuir(2, 2) = .True. 
      Langmuir(2, 1) = .True. 
      Pow(2, 2) = 1.0d0 
      Pow(2, 1) = 1.0d0 
      Nimax(2, 2) = 0.026185d0 
      Nimax(2, 1) = 0.696033d0 
      Ki(2, 2) = 0.00000000000d0 
      Ki(2, 1) = 1.11205024000d0 
      Yi(1) =  0.50d0 
      Langmuir(1, 2) = .True. 
      Langmuir(1, 1) = .True. 
      Pow(1, 2) = 1.0d0 
      Pow(1, 1) = 1.0d0 
      Nimax(1, 2) = 0.225624d0 
      Nimax(1, 1) = 0.467340d0 
      Ki(1, 2) = 0.00003254606d0 
      Ki(1, 1) = 0.00003254683d0 
C     End for Python1


      Carrier_gas = 1       
      P   = 1.0d4
      Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
C     Start for Python2
      write(25,'(A)') "  Pressure (Pa) 23mC5-500 (mol/kg) 24mC5-500 (mol/kg) 2mC6-500 (mol/kg) 3mC6-500 (mol/kg) C7-500 (mol/kg)" 
      write(6,'(2e20.10)') Ni(1),Ni(2),Ni(3),Ni(4),Ni(5)
      write(6,*) 'Ni(1)   ','Ni(2)   ','Ni(3)   ','Ni(4)   ','Ni(5)   '
C     End for Python2
C      stop

C      stop

      Do J = 0, 8
         If(J.Lt.8) Then
            Do I = 1, 50, 2
               P = Dble(I*10**J)
               Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
C     Start for Python3
                Write(25,'(20e20.10)') P,Ni(1),Ni(2),Ni(3),Ni(4),Ni(5)
            Enddo
         Else
            P = Dble(1.0*10**J)
            Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
C     Start for Python4
            Write(25,'(20e20.10)') P,Ni(1),Ni(2),Ni(3),Ni(4),Ni(5)
         Endif
      Enddo
      
      
      Stop
      End
