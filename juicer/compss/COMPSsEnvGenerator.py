# coding=utf-8
import sys
import tarfile
import os

def make_tarfile(output, app, project, resources, lib):
    with tarfile.open(output, "w:gz") as tar:
        for name in [app, project, resources, lib]:
            tar.add(name, arcname=os.path.basename(name))

def generateProject(settings, out=None):

    InstanceType = ''
    for instance in settings['instances']:
        InstanceType += '<InstanceType Name="{}" /> '.format(instance)

    project = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Project>
    <MasterNode />
    <Cloud>
        <InitialVMs>1</InitialVMs>
        <MinimumVMs>{MinimumVMs}</MinimumVMs>
        <MaximumVMs>{MaximumVMs}</MaximumVMs>

        <CloudProvider Name="Lemonade">
            <Properties>
                <!--  network and auth -->
                <Property>
                 <Name>mesos-default-principal</Name>
                 <Value>ubuntu</Value>
                </Property>

                <Property>
                 <Name>mesos-default-secret</Name>
                 <Value>ubuntusecret</Value>
                </Property>

                <Property>
                 <Name>mesos-docker-network</Name>
                 <Value>bigsea-net</Value>
                </Property>

                <!-- Mesos settings -->
                <Property>
                 <Name>mesos-framework-register-timeout</Name>
                 <Value>30000</Value>
                </Property>

                <Property>
                 <Name>mesos-authenticate</Name>
                 <Value>true</Value>
                </Property>

                <!-- Optional connector parameters -->
                <Property>
                 <Name>max-vm-creation-time</Name>
                 <Value>10</Value>
                </Property> <!-- Minutes -->

                <Property>
                 <Name>max-connection-errors</Name>
                 <Value>36</Value>
                </Property>

                <!-- Abstract SSH Connector parameters -->
                <Property>
                 <Name>vm-user</Name> <Value>root</Value> </Property>

                <Property>
                 <Name>vm-keypair-name</Name>
                 <Value>id_rsa</Value>
                </Property>

                <Property>
                 <Name>vm-keypair-location</Name>
                 <Value>/root/.ssh/</Value>
                </Property>
            </Properties>

            <Images>
                <Image Name="{image}">
                    <InstallDir>/opt/COMPSs/</InstallDir>
                    <WorkingDir>/root/</WorkingDir>
                    <User>root</User>
                        <Application> <AppDir>/root/</AppDir> </Application>
                    <Package>
                        <Source>/root/{Application}</Source>
                        <Target>/root</Target>
                    </Package>
                </Image>
            </Images>

            <InstanceTypes>
                {InstanceType}
            </InstanceTypes>
        </CloudProvider>
    </Cloud>
</Project>""".format(MinimumVMs=settings['MinimumVMs'],
                     MaximumVMs=settings['MaximumVMs'],
                     image=settings['image'],
                     Application=settings['Application'],
                     InstanceType=InstanceType)

    if out is None:
        sys.stdout.write(project.encode('utf8'))
    else:
        out.write(project)

def generateResources(settings, out=None):
    resources = """
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<ResourcesList>
    <CloudProvider Name="Lemonade">
        <Endpoint>
            <Server>zk://localhost:2181/mesos</Server>
            <ConnectorJar>mesos-conn.jar</ConnectorJar>
            <ConnectorClass>es.bsc.conn.mesos.Mesos</ConnectorClass>
        </Endpoint>

        <Images>
            <Image Name="{image}">
                <!-- Maximum VM creation time (in seconds) -->
                <CreationTime>60</CreationTime>
                <Adaptors>
                    <Adaptor Name="integratedtoolkit.nio.master.NIOAdaptor">
                        <SubmissionSystem> <Interactive /> </SubmissionSystem>
                        <Ports>
                            <MinPort>43100</MinPort>
                            <MaxPort>43105</MaxPort>
                        </Ports>
                    </Adaptor>
                    <Adaptor Name="integratedtoolkit.gat.master.GATAdaptor">
                        <SubmissionSystem>
                            <Batch> <Queue>sequential</Queue> </Batch>
                            <Interactive />
                        </SubmissionSystem>
                        <BrokerAdaptor>sshtrilead</BrokerAdaptor>
                    Adaptor>
                </Adaptors>
            </Image>
        </Images>

        <InstanceTypes>
            <InstanceType Name="small">
                <Processor Name="Processor1">
                 <ComputingUnits>2</ComputingUnits>
                </Processor>
                <Memory> <Size>0.5</Size> </Memory>
                <Storage> <Size>5.0</Size> </Storage>
                </Price>
            </InstanceType>

            <InstanceType Name="medium">
                <Processor Name="Processor1">
                 <ComputingUnits>4</ComputingUnits>
                </Processor>
                <Memory> <Size>1.0</Size> </Memory>
                <Storage> <Size>10.0</Size> </Storage>
                <Price> <TimeUnit>1</TimeUnit>
                 <PricePerUnit>0.212</PricePerUnit> </Price>
            </InstanceType>

            <InstanceType Name="large">
                <Processor Name="Processor1">
                 <ComputingUnits>8</ComputingUnits>
                </Processor>
                <Memory> <Size>4</Size> </Memory>
                <Storage> <Size>10.0</Size> </Storage>
                </Price>
            </InstanceType>
            <InstanceType Name="extra_large">
                <Processor Name="Processor1">
                 <ComputingUnits>8</ComputingUnits>
                </Processor>
                <Memory> <Size>8</Size> </Memory>
                <Storage> <Size>10.0</Size> </Storage>
            </InstanceType>
        </InstanceTypes>
    </CloudProvider>
</ResourcesList>
""".format(image=settings['image'])

    if out is None:
        sys.stdout.write(resources.encode('utf8'))
    else:
        out.write(resources)
