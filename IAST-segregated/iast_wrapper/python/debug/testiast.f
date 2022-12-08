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
      Ncomp = 3 
      Yi(3) =  0.90d0 
      Langmuir(3, 2) = .True. 
      Langmuir(3, 1) = .True. 
      Pow(3, 2) = 1.0d0 
      Pow(3, 1) = 1.0d0 
      Nimax(3, 2) = 0.483846d0 
      Nimax(3, 1) = 0.686442d0 
      Ki(3, 2) = 0.00001472505d0 
      Ki(3, 1) = 0.05036542620d0 
      Yi(2) =  0.10d0 
      Langmuir(2, 2) = .True. 
      Langmuir(2, 1) = .True. 
      Pow(2, 2) = 1.0d0 
      Pow(2, 1) = 1.0d0 
      Nimax(2, 2) = 0.234467d0 
      Nimax(2, 1) = 0.461372d0 
      Ki(2, 2) = 0.02310090000d0 
      Ki(2, 1) = 0.02310055000d0 
      Yi(1) =  0.00d0 
      Langmuir(1, 2) = .True. 
      Langmuir(1, 1) = .True. 
      Pow(1, 2) = 1.0d0 
      Pow(1, 1) = 1.0d0 
      Nimax(1, 2) = 0.680083d0 
      Nimax(1, 1) = 0.815592d0 
      Ki(1, 2) = 0.01566802000d0 
      Ki(1, 1) = 0.00000000305d0 
C     End for Python1


      Carrier_gas = 1       
      P   = 1.0d4
      Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
C     Start for Python2
      write(25,'(A)')   '     Pressure (Pa)     22mC5 -400 (mol/kg)     2mC6 -400 (mol/kg)     C7 -400 (mol/kg)'
      write(6,'(2e20.10)') Ni(1),Ni(2),Ni(3)
      write(6,*) 'Ni(1)   ','Ni(2)   ',Ni(3)   

C     End for Python2   
C      stop

C      stop

      Do J = 0, 8
         If(J.Lt.8) Then
            Do I = 1, 3, 2
               P = Dble(I*10**J)
               Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
               Write(25,'(5e20.10)') P,Ni(2),Ni(3)
            Enddo
         Else
            P = Dble(1.0*10**J)
            Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
            Write(25,'(5e20.10)') P,Ni(2),Ni(3)
         Endif
      Enddo
      
      
      Stop
      End
      
