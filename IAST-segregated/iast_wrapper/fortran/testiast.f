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
      Ncomp = 4 
      Yi(4) =  0.55d0 
      Langmuir(4, 2) = .True. 
      Langmuir(4, 1) = .True. 
      Pow(4, 2) = 1.0d0 
      Pow(4, 1) = 1.0d0 
      Nimax(4, 2) = 0.216267d0 
      Nimax(4, 1) = 0.467867d0 
      Ki(4, 2) = 0.00000165003d0 
      Ki(4, 1) = 0.00000165006d0 
      Yi(3) =  0.20d0 
      Langmuir(3, 2) = .True. 
      Langmuir(3, 1) = .True. 
      Pow(3, 2) = 1.0d0 
      Pow(3, 1) = 1.0d0 
      Nimax(3, 2) = 0.083818d0 
      Nimax(3, 1) = 0.657759d0 
      Ki(3, 2) = 0.00000002425d0 
      Ki(3, 1) = 0.00000195937d0 
      Yi(2) =  0.10d0 
      Langmuir(2, 2) = .True. 
      Langmuir(2, 1) = .True. 
      Pow(2, 2) = 1.0d0 
      Pow(2, 1) = 1.0d0 
      Nimax(2, 2) = 1849.802590d0 
      Nimax(2, 1) = 0.656814d0 
      Ki(2, 2) = 0.00000000000d0 
      Ki(2, 1) = 0.00000201664d0 
      Yi(1) =  0.15d0 
      Langmuir(1, 2) = .True. 
      Langmuir(1, 1) = .True. 
      Pow(1, 2) = 1.0d0 
      Pow(1, 1) = 1.0d0 
      Nimax(1, 2) = 0.214660d0 
      Nimax(1, 1) = 0.478733d0 
      Ki(1, 2) = 0.00000427492d0 
      Ki(1, 1) = 0.00000427493d0 
C     End for Python1


      Carrier_gas = 1       
      P   = 1.0d4
      Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
C     Start for Python2
      write(25,'(A)') "  Pressure (Pa) @ 600K 3eC5 (mol/kg) 33mC5 (mol/kg) 23mC5 (mol/kg) 2mC4 (mol/kg)" 
      write(6,'(2e20.10)') Ni(1),Ni(2),Ni(3),Ni(4)
      write(6,*) 'Ni(1)   ','Ni(2)   ','Ni(3)   ','Ni(4)   '
C     End for Python2
C      stop

C      stop

      Do J = 0, 8
         If(J.Lt.8) Then
            Do I = 1, 20, 2
               P = Dble(I*10**J)
               Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
C     Start for Python3
                Write(25,'(20e20.10)') P,Ni(1),Ni(2),Ni(3),Ni(4)
            Enddo
         Else
            P = Dble(1.0*10**J)
            Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
C     Start for Python4
            Write(25,'(20e20.10)') P,Ni(1),Ni(2),Ni(3),Ni(4)
         Endif
      Enddo
      
      
      Stop
      End
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
      Ncomp = 4 
      Yi(4) =  0.55d0 
      Langmuir(4, 2) = .True. 
      Langmuir(4, 1) = .True. 
      Pow(4, 2) = 1.0d0 
      Pow(4, 1) = 1.0d0 
      Nimax(4, 2) = 0.073920d0 
      Nimax(4, 1) = 0.615449d0 
      Ki(4, 2) = 0.00004527824d0 
      Ki(4, 1) = 0.00001544620d0 
      Yi(3) =  0.15d0 
      Langmuir(3, 2) = .True. 
      Langmuir(3, 1) = .True. 
      Pow(3, 2) = 1.0d0 
      Pow(3, 1) = 1.0d0 
      Nimax(3, 2) = 0.216267d0 
      Nimax(3, 1) = 0.467867d0 
      Ki(3, 2) = 0.00000165003d0 
      Ki(3, 1) = 0.00000165006d0 
      Yi(2) =  0.20d0 
      Langmuir(2, 2) = .True. 
      Langmuir(2, 1) = .True. 
      Pow(2, 2) = 1.0d0 
      Pow(2, 1) = 1.0d0 
      Nimax(2, 2) = 0.068996d0 
      Nimax(2, 1) = 0.656710d0 
      Ki(2, 2) = 266.99638600000d0 
      Ki(2, 1) = 0.00000139356d0 
      Yi(1) =  0.10d0 
      Langmuir(1, 2) = .True. 
      Langmuir(1, 1) = .True. 
      Pow(1, 2) = 1.0d0 
      Pow(1, 1) = 1.0d0 
      Nimax(1, 2) = 0.214660d0 
      Nimax(1, 1) = 0.478733d0 
      Ki(1, 2) = 0.00000427492d0 
      Ki(1, 1) = 0.00000427493d0 
C     End for Python1


      Carrier_gas = 1       
      P   = 1.0d4
      Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
C     Start for Python2
      write(25,'(A)') "  Pressure (Pa) @ 600K 3eC5 (mol/kg) 23mC4 (mol/kg) 2mC4 (mol/kg) nC7 (mol/kg)" 
      write(6,'(2e20.10)') Ni(1),Ni(2),Ni(3),Ni(4)
      write(6,*) 'Ni(1)   ','Ni(2)   ','Ni(3)   ','Ni(4)   '
C     End for Python2
C      stop

C      stop

      Do J = 0, 8
         If(J.Lt.8) Then
            Do I = 1, 20, 2
               P = Dble(I*10**J)
               Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
C     Start for Python3
                Write(25,'(20e20.10)') P,Ni(1),Ni(2),Ni(3),Ni(4)
            Enddo
         Else
            P = Dble(1.0*10**J)
            Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
C     Start for Python4
            Write(25,'(20e20.10)') P,Ni(1),Ni(2),Ni(3),Ni(4)
         Endif
      Enddo
      
      
      Stop
      End
